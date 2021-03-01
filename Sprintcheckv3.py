from jira import JIRA
import yaml
import json
import pandas as pd
import json
import sys
from datetime import datetime, date
import re
import numpy as np


def read_yaml(fname):
    try:
        with open(fname, "rt", encoding="utf8") as file:
            Od = yaml.load(file, Loader=yaml.FullLoader)
            print(fname + " file read!")
            return Od
    except:
        print("Failed reading file " + fname + "!")


def update_field_mappings(jira_conn, fname):
    mappings = read_yaml(fname)
    allfields = jira_conn.fields()
    update = {}
    # Make a map from field name -> field id
    nameMap = {field["name"]: field["id"] for field in allfields}
    for i, j in nameMap.items():
        if i not in mappings.keys():
            if "fields." + str(j) != mappings.get(i):
                update[i] = "fields." + str(j)
    if len(update) != 0:
        with open(fname, "a") as f:
            f.write("\n")
            yaml.dump(update, f, encoding="utf8")
    else:
        print("mappings.yaml already updated..")


def Jira_conn(user, apikey, server):
    options = {"server": server}
    try:
        # jira_conn = JIRA('https://jira.atlassian.com')
        jira_conn = JIRA(options, basic_auth=(user, apikey))
        print("Connection to jira established!")
    except:
        print("Connection to jira failed!Check credentials")
    return jira_conn


def get_projects_in_server(jira_conn):
    project_list = []
    for i in jira_conn.projects():
        project_list.append(i.name)
    return project_list


def get_all_issues(jira, proj, jql, filter=0):
    block_size = 100
    block_num = 0
    allissues = []
    while True:
        issues = []
        start_idx = block_num * block_size
        if len(jql) != 0:
            issues = jira.search_issues(
                "project= " + proj + " AND " + jql,
                start_idx,
                block_size,
                json_result=True,
            )
        elif filter != 0:
            issues = jira.search_issues(
                "filter=" + str(filter), start_idx, block_size, json_result=True
            )
        else:
            issues = jira.search_issues(
                "project=" + proj, start_idx, block_size, json_result=True
            )
        if len(issues["issues"]) == 0:
            # Retrieve issues until there are no more to come
            break
        block_num += 1
        for issue in issues["issues"]:
            allissues.append(issue)
    return allissues


def rename_cols(df, fields, exclude_cols):
    df = df
    mappings = read_yaml("mappings.yaml")
    d = {}
    if len(exclude_cols) != 0:
        print("Excluding columns....")
        for i in exclude_cols:
            l = [c for c in df.columns if re.search(i.lower(), c.lower()) == None]
            df = df[l]
    for col in list(df.columns):
        for i, j in mappings.items():
            if str(j) == str(col):
                d[str(col)] = str(i)
            elif re.search(str(j).lower(), str(col).lower()) != None:
                d[str(col)] = re.sub(str(j), str(i), str(col))
    df = df.rename(columns=d)
    return df


def get_issues(jira_conn, projects, jql, fname, fields, exclude_cols, filters=0):
    final_df = pd.DataFrame()
    if filters != 0:
        final_df = execute_filter(jira_conn, fname, exclude_cols, fields, filters)
    else:
        for project in projects:
            print("Fetching issues for project=" + project + ".......")
            allissues_details = get_all_issues(jira_conn, project, jql)
            issue_df = pd.json_normalize(allissues_details)
            issue_df = rename_cols(issue_df, fields, exclude_cols)
            final_df = pd.concat([final_df, issue_df], axis=0, ignore_index=True)
    return final_df


def createTempDF(df):
    print("Creating temporary df.....")
    df = df.fillna("")
    # filter out backlog issues
    df = df[df["Sprint"] != ""]
    df2_vals = []
    for i, j, k in zip(df["Issue id"], df["Sprint"], df["Project.id"]):
        for z in j:
            x = []
            if (
                ("completeDate" in z.keys())
                & ("startDate" in z.keys())
                & ("endDate" in z.keys())
            ):
                x = [
                    i,
                    z["id"],
                    k,
                    z["name"],
                    z["state"],
                    z["startDate"],
                    z["endDate"],
                    z["completeDate"],
                ]
            elif (
                ("completeDate" not in z.keys())
                & ("startDate" in z.keys())
                & ("endDate" in z.keys())
            ):
                x = [
                    i,
                    z["id"],
                    k,
                    z["name"],
                    z["state"],
                    z["startDate"],
                    z["endDate"],
                    "",
                ]
            else:
                x = [i, z["id"], k, z["name"], z["state"], "", "", ""]
            df2_vals.append(x)

    df2 = pd.DataFrame(
        df2_vals,
        columns=[
            "Issue id",
            "Sprint id",
            "Project.id",
            "Sprint name",
            "Sprint state",
            "Sprint startDate",
            "Sprint endDate",
            "Sprint completeDate",
        ],
    )
    df3 = df.merge(df2, on=["Issue id", "Project.id"])
    df3 = df3.drop(["Sprint"], axis=1)
    df3 = df3.drop_duplicates()
    return df3


def IssueInNextSprint(sprintID, issueId, df):
    if (
        df[(df["Sprint id"] == int(sprintID) + 1) & (df["Issue id"] == issueId)].shape[
            0
        ]
        == 1
    ):
        return True
    else:
        return False


def MovedToNextSprint(df_org, df, colname, status):
    if len(status) == 1:
        df = df[(df["Issue Type.name"] == "Story") & (df["Status.name"] == status[0])]
    elif len(status) == 2:
        df = df[
            (df["Issue Type.name"] == "Story")
            & ((df["Status.name"] == status[0]) | (df["Status.name"] == status[1]))
        ]
    if df.shape[0] != 0:
        op_df = pd.DataFrame()
        df["Is in  next sprint"] = df.apply(
            lambda x: IssueInNextSprint(x["Sprint id"], x["Issue id"], df), axis=1
        )
        op_df = pd.DataFrame(
            df[df["Is in  next sprint"] == True].groupby(["Sprint id"]).size(),
            columns=[colname],
        )
        op_df = df_org.merge(op_df, on="Sprint id", how="left")
        return op_df
    else:
        return df_org


# --------------------------------------------------------------------------------------------------------------------------
def IssueInPreviousSprint(sprintID, issueId, df):
    if (
        df[(df["Sprint id"] == int(sprintID) - 1) & (df["Issue id"] == issueId)].shape[
            0
        ]
        == 1
    ):
        return True
    else:
        return False


def MovedFromPreviousSprint(df_org, df, colname):
    df = df[df["Issue Type.name"] == "Story"]
    op_df = pd.DataFrame()
    if df.shape[0] != 0:
        df["Is in  previous sprint"] = df.apply(
            lambda x: IssueInPreviousSprint(x["Sprint id"], x["Issue id"], df), axis=1
        )
        # print(df[['Issue id','Project.key','Sprint id','Is in  previous sprint']])
        op_df = pd.DataFrame(
            df[df["Is in  previous sprint"] == True].groupby(["Sprint id"]).size(),
            columns=[colname],
        )
        op_df = df_org.merge(op_df, on="Sprint id", how="left")
        return op_df
    else:
        return df_org


# --------------------------------------------------------------------------------------------------------------------------
def getCount(df_org, df, colname):
    if df.shape[0] != 0:
        op_df = df.groupby(["Sprint id"]).count()[["Issue id"]]
        op_df.columns = [colname]
        op_df = df_org.merge(op_df, on=["Sprint id"], how="left")
        return op_df
    else:
        return df_org


def userStoriesCommited(df_org, df, colname):
    df = df[df["Issue Type.name"] == "Story"]
    if df.shape[0] != 0:
        return getCount(df_org, df, colname)
    else:
        return df_org


# --------------------------------------------------------------------------------------------------------------------------
def issuesBasedOnStatus(
    df_org, df, colname, pattern="", status=[], issueType=""
):  # pattern mainly for Summary&labels
    if issueType != "":
        df = df[df["Issue Type.name"] == issueType]
    if (df.shape[0] != 0) & (pattern != ""):
        df = filterColumnOnPattern(df, pattern)
    if (df.shape[0] != 0) & (len(status) != 0):
        if len(status) == 2:
            df = df[(df["Status.name"] == status[0]) | (df["Status.name"] == status[1])]
        elif len(status) == 1:
            df = df[df["Status.name"] == status[0]]
        return getCount(df_org, df, colname)
    else:
        return df_org


# --------------------------------------------------------------------------------------------------------------------------
def issuesBasedOnValuePattern(
    df_org, df, colname, pattern="", issueType=""
):  # checks if pattern in Summary or Labels
    if issueType != "":
        df = df[df["Issue Type.name"] == issueType]
    if (df.shape[0] != 0) & (pattern != ""):
        df = filterColumnOnPattern(df, pattern)
        # case sesnitive!
        return getCount(df_org, df, colname)
    else:
        return df_org


# --------------------------------------------------------------------------------------------------------------------------
def jqlBuilder(change):
    jql = ""
    for i in range(0, len(change) - 1):
        if i < len(change) - 1:
            jql = (
                jql
                + "status CHANGED FROM '"
                + change[i]
                + "' TO '"
                + change[i + 1]
                + "' AND "
            )
    return jql[:-5]


def issueStatusChange(jira_conn, df, change, proj):
    allissues_details = []
    issue_df = pd.DataFrame()
    for i in change:
        jql = jqlBuilder(i)
        issues_list = get_all_issues(jira_conn, proj, jql, filter=0)
        allissues_details.extend(issues_list)

    if len(allissues_details) != 0:
        issue_df = pd.json_normalize(allissues_details)
        op_df = issue_df[["key"]]
        op_df.columns = ["Issue key"]
        df = df.merge(op_df, on=["Issue key"])
        return df
    else:
        return issue_df


def issuesBasedOnStatusChange(
    df_org, df, colname, change, proj, jira_conn, issueType="", pattern=""
):
    if issueType != "":
        df = df[df["Issue Type.name"] == issueType]
    if (df.shape[0] != 0) & (pattern != ""):
        df = filterColumnOnPattern(df, pattern)
    df = issueStatusChange(jira_conn, df, change, proj)
    if df.shape[0] != 0:
        return getCount(df_org, df, colname)
    else:
        return df_org


# --------------------------------------------------------------------------------------------------------------------------
def getPriorityCount(df_org, df, colname, pattern, priority="", issueType=""):
    if issueType != "":
        df = df[df["Issue Type.name"] == issueType]
    if (df.shape[0] != 0) & (pattern != ""):
        df = filterColumnOnPattern(df, pattern)
    if df.shape[0] != 0:
        if priority == "Medium":
            df = df[df["Priority.name"] == "Medium"]
        elif priority == "Low":
            df = df[(df["Priority.name"] == "Low") | (df["Priority.name"] == "Lowest")]
        elif priority == "High":
            df = df[
                (df["Priority.name"] == "High") | (df["Priority.name"] == "Highest")
            ]
        # case sesnitive!
        return getCount(df_org, df, colname)
    else:
        return df_org


def check_condition(col, linkedTo, status, how):
    op = False
    if len(col) != 0:
        for i in col:
            if status != "":
                if (i[how]["fields"]["status"]["name"] != status) & (
                    i[how]["fields"]["issuetype"]["name"] == linkedTo
                ):
                    op = True
                    break
            else:
                if i[how]["fields"]["issuetype"]["name"] == linkedTo:
                    op = True
                    break
    return op


def getLinks(df, linkedIn, linkedTo, how="inward", status="", colname=""):
    df_linkedIssue = df[["Issue id", "Issue Type.name", "Linked Issues"]]
    df_linkedIssue = df_linkedIssue[df_linkedIssue["Issue Type.name"] == linkedIn]
    if df_linkedIssue.shape[0] != 0:
        df_linkedIssue[colname] = df_linkedIssue.apply(
            lambda x: check_condition(x["Linked Issues"], linkedTo, status, how), axis=1
        )
        df_linkedIssue.drop(["Linked Issues", "Issue Type.name"], axis=1, inplace=True)
        df = df.merge(df_linkedIssue, on="Issue id", how="left")
    return df


def filterColumnOnValue(df, colname, value):
    df = df[df[colname] == value]
    return df


def filterColumnOnPattern(df, pattern):
    df["matches"] = df.apply(
        lambda x: True
        if (re.search(pattern.lower(), x["Summary"].lower())) != None
        else (
            True
            if (re.search(pattern.lower(), str(x["Labels"]).lower())) != None
            else False
        ),
        axis=1,
    )
    return df[df["matches"] == True].drop(["matches"], axis=1)


def getChangeLogForIssue(jira_conn, df, change, issueType=""):
    # supports 2 changes
    op_df = pd.DataFrame(
        columns=[
            "Issue key",
            "In Progress",
            "In Testing",
            "Ready for Acceptance",
            "Rejected",
            "Re-open",
            "Done",
        ]
    )
    if issueType != "":
        df = df[df["Issue Type.name"] == issueType]
    if df.shape[0] != 0:
        issueKeys = list(df["Issue key"])
        for i in issueKeys:
            d = {}
            d["Issue key"] = i
            issue = jira_conn.issue(i, expand="changelog")
            changelog = issue.changelog
            for history in changelog.histories:
                for item in history.items:
                    if item.field == "status":
                        if len(change) == 1:
                            if (item.fromString == change[0][0]) & (
                                item.toString == change[0][1]
                            ):
                                if item.toString not in d.keys():
                                    d[item.toString] = history.created
                            elif item.toString == change[0][0]:
                                if item.toString not in d.keys():
                                    d[item.toString] = history.created
                        elif len(change) == 2:
                            if (
                                (item.fromString == change[0][0])
                                & (item.toString == change[0][1])
                            ) | (
                                (item.fromString == change[1][0])
                                & (item.toString == change[1][1])
                            ):
                                if item.toString not in d.keys():
                                    d[item.toString] = history.created
                            elif (item.toString == change[0][0]) | (
                                item.toString == change[1][0]
                            ):
                                if item.toString not in d.keys():
                                    d[item.toString] = history.created
            if len(d) >= 2:
                op_df = op_df.append(d, ignore_index=True).fillna("")
        for i in [
            "In Progress",
            "In Testing",
            "Ready for Acceptance",
            "Rejected",
            "Re-open",
            "Done",
        ]:
            op_df[i] = op_df.apply(lambda x: pd.Timestamp(x[i]), axis=1)
    return op_df.drop_duplicates().fillna("")


def create_reports():
    conf = read_yaml("config2.yaml")
    user = conf["user"]
    apikey = conf["apikey"]
    server = conf["server"]
    date_range = conf["date_range"]
    sprints = conf["sprints"]
    jira_conn = Jira_conn(user, apikey, server)
    df = pd.DataFrame()
    fname = conf["op_filename"]
    exclude_cols = conf["exclude_field_val"]
    # jql= conf['jql']
    filters = 0
    # fields= conf['fields']
    projects = conf["projects"]
    if len(projects) == 0:
        projects = get_projects_in_server(jira_conn)
    if conf["update_field_mappings"] == True:
        update_fields = update_field_mappings(jira_conn, "mappings.yaml")
    df = get_issues(jira_conn, projects, "", fname, [], exclude_cols, filters)
    df = df[
        [
            "Issue key",
            "Issue id",
            "Issue Type.name",
            "Summary",
            "Project.id",
            "Project.key",
            "Project.name",
            "Priority.name",
            "Project.projectTypeKey",
            "Status Category Changed",
            "Status.description",
            "Status.name",
            "Status.statusCategory.name",
            "Status Category Changed",
            "Sprint",
            "Labels",
            "Due Date",
            "Linked Issues",
        ]
    ]

    df = getLinks(df, "Story", "Bug", "inwardIssue", "Done", "OpenDefectsLinkedToStory")
    df = getLinks(df, "Bug", "Story", "outwardIssue", "", "BugsLinkedToUserStory")
    df.drop(["Linked Issues"], axis=1, inplace=True)
    # since drop_duplicates doesnot work with list columns
    df["Labels"] = df["Labels"].astype("str")
    df_temp = createTempDF(df)
    sprint_df = df_temp[
        [
            "Project.name",
            "Project.id",
            "Project.key",
            "Project.projectTypeKey",
            "Sprint id",
            "Sprint name",
            "Sprint state",
            "Sprint startDate",
            "Sprint endDate",
            "Sprint completeDate",
        ]
    ].drop_duplicates()
    final_df = pd.DataFrame()
    print("Generating desired fields......")
    for i in list(sprint_df["Project.key"].unique()):
        df_temp2 = df_temp[df_temp["Project.key"] == i]
        op = userStoriesCommited(
            sprint_df[sprint_df["Project.key"] == i],
            df_temp2,
            "# User stories commited",
        )

        op = MovedFromPreviousSprint(
            op, df_temp2, "# User Storied Carried From Previous Sprint"
        )

        op = MovedToNextSprint(
            op, df_temp2, "# User Stories Spilled To Next Sprint (IQE)", ["In Testing"]
        )

        op = MovedToNextSprint(
            op,
            df_temp2,
            "# User Stories Spilled To Next Sprint (Dev)",
            ["In Progress", "To Do"],
        )

        # user stories delivered on time to IQE
        df_temp9 = df_temp2[df_temp2["Issue Type.name"] == "Story"]
        if df_temp9.shape[0] != 0:
            df_temp9 = issueStatusChange(
                jira_conn, df_temp9, [["In Progress", "In Testing"]], i
            )
        if df_temp9.shape[0] != 0:
            issueChangeLog = getChangeLogForIssue(
                jira_conn, df_temp9, [["In Progress", "In Testing"]], issueType=""
            )
            # since due date doesnt have time componenet,normalize time component from other end
            issueChangeLog["In Testing"] = issueChangeLog["In Testing"].dt.normalize()
            # convert due date column(string) to timestamp
            df_temp9["Due Date"] = df_temp9.apply(
                lambda x: pd.Timestamp(x["Due Date"]), axis=1
            )
            # remove timezone
            df_temp9["Due Date"] = df_temp9["Due Date"].dt.tz_localize(None)
            df_temp9 = df_temp9[["Sprint id", "Issue key", "Due Date"]].merge(
                issueChangeLog[["Issue key", "In Testing"]], on=["Issue key"]
            )
            df_temp9["In Testing"] = df_temp9["In Testing"].dt.tz_localize(None)
            # df_temp9['#user stories delivered on time to IQE'] = df_temp9[['Issue key','Due Date','In Testing']].apply(lambda x: calc(x['In Testing'],x['In Testing']))
            df_temp9["Actual Time"] = df_temp9.apply(
                lambda x: (x["Due Date"] - x["In Testing"])
                if ((x["Due Date"] != "") & (x["In Testing"] != ""))
                else 0,
                axis=1,
            )
            if df_temp9.shape[0] != 0:
                df_temp9["# User stories delivered on time to IQE"] = df_temp9.apply(
                    lambda x: True
                    if x["Actual Time"] / np.timedelta64(1, "h") >= 0
                    else False,
                    axis=1,
                )
            # filter where'#User stories delivered on time to IQE' = True
            df_temp9 = df_temp9[
                df_temp9["# User stories delivered on time to IQE"] == True
            ]
            if df_temp9.shape[0] != 0:
                x = df_temp9.groupby(["Sprint id"]).size()
                df_temp9 = pd.DataFrame(
                    x, columns=["# User stories delivered on time to IQE"]
                )
                op = op.merge(df_temp9, on=["Sprint id"], how="left")

        op = issuesBasedOnStatusChange(
            op,
            df_temp2,
            "# User Stories Delivered for IQE to Test",
            [["In Progress", "In Testing"]],
            i,
            jira_conn,
            "Story",
            "",
        )

        df_temp3 = filterColumnOnValue(df_temp2, "OpenDefectsLinkedToStory", True)
        if df_temp3.shape[0] != 0:  # upate
            op = issuesBasedOnStatusChange(
                op,
                df_temp3,
                "# User Stories Completed by IQE",
                [["In Progress", "In Testing", "Ready For Acceptance"]],
                i,
                jira_conn,
                "Story",
                "",
            )

        op = issuesBasedOnStatusChange(
            op,
            df_temp2,
            "# User Stories Certified by IQE",
            [["In Testing", "Ready for Acceptance"]],
            i,
            jira_conn,
            "Story",
            "",
        )

        op = issuesBasedOnStatusChange(
            op,
            df_temp2,
            "# User Stories Rejected by IQE",
            [["In Testing", "Rejected"]],
            i,
            jira_conn,
            "Story",
            "",
        )

        op = issuesBasedOnStatusChange(
            op,
            df_temp2,
            "# User Stories Completed",
            [["To Do", "In Testing", "Ready for Acceptance", "Done"]],
            i,
            jira_conn,
            "Story",
            "",
        )

        op = issuesBasedOnStatusChange(
            op,
            df_temp2,
            "# User Stories Passed by IQE but Rejected by PO",
            [["Ready for Acceptance", "Rejected"]],
            i,
            jira_conn,
            "Story",
            "",
        )

        df_temp4 = filterColumnOnValue(df_temp, "BugsLinkedToUserStory", True)
        if df_temp4.shape[0] != 0:
            op = getCount(op, df_temp4, "# of Defects Logged by IQE (In-Sprint)")

        df_temp5 = df_temp2[df_temp2["BugsLinkedToUserStory"] == True]
        if df_temp5.shape[0] != 0:
            df_temp5.loc[
                df_temp5["Status.name"] == "Done", "BugsLinkedToUserStory"
            ] = False
        df_temp51 = filterColumnOnPattern(df_temp2, "^reg")
        df_temp5 = pd.concat(
            [df_temp5, df_temp51], axis=0, ignore_index=True
        ).drop_duplicates()
        if df_temp5.shape[0] != 0:
            op = issuesBasedOnStatus(
                op,
                df_temp5,
                "Defects Re-tested (In-Sprint + Regression)",
                "",
                ["Re-open"],
                "Bug",
            )
        # look into this
        if df_temp5.shape[0] != 0:
            op = issuesBasedOnStatusChange(
                op,
                df_temp5,
                "# Defects Rejected (In-Sprint)",
                [["In Testing", "Rejected"]],
                i,
                jira_conn,
                "Bug",
                "",
            )

        op = issuesBasedOnStatus(
            op,
            df_temp2,
            "Defects Slipped by Dev Team",
            "",
            ["To Do", "In Progress"],
            "Bug",
        )

        op = issuesBasedOnValuePattern(
            op,
            df_temp2,
            "# of Defects Logged from Regression Execution)",
            "^reg",
            "Bug",
        )

        # reg
        op = issuesBasedOnStatus(
            op, df_temp2, "# Defects Open", "^reg", ["In Progress"], "Bug"
        )

        # reg
        op = issuesBasedOnStatus(
            op,
            df_temp2,
            "# Defects Closed",
            "^reg",
            ["Done", "Ready for Acceptance"],
            "Bug",
        )

        # Reg
        op = issuesBasedOnStatus(
            op, df_temp2, "# Defects Rejected", "^reg", ["Rejected"], "Bug"
        )

        op = issuesBasedOnValuePattern(
            op, df_temp2, "# of Defects reported in UAT", "^uat", "Bug"
        )

        op = issuesBasedOnStatus(
            op, df_temp2, "Re-opened Defects Count(UAT)", "^uat", ["Re-open"], "Bug"
        )

        op = getPriorityCount(
            op, df_temp2, "High Priority(UAT defects)", "^UAT", "High", "Bug"
        )
        op = getPriorityCount(
            op, df_temp2, "Medium Priority(UAT defects)", "^UAT", "Medium", "Bug"
        )
        op = getPriorityCount(
            op, df_temp2, "Low Priority(UAT defects)", "^UAT", "Low", "Bug"
        )

        # IQE turn around time
        df_temp7 = df_temp2[df_temp2["Issue Type.name"] == "Bug"]
        if df_temp7.shape[0] != 0:
            df_temp7 = issueStatusChange(
                jira_conn,
                df_temp7,
                [["In Testing", "Ready for Acceptance"], ["In Testing", "Done"]],
                i,
            )
        if df_temp7.shape[0] != 0:
            issueChangeLog = getChangeLogForIssue(
                jira_conn,
                df_temp7,
                [["In Testing", "Ready for Acceptance"], ["In Testing", "Done"]],
                issueType="",
            )
            # nat =np.datetime64('NaT')
            issueChangeLog["IQE Turn Around Time"] = issueChangeLog.apply(
                lambda x: (x["Ready for Acceptance"] - x["In Testing"])
                if ((x["In Testing"] != "") & (x["Ready for Acceptance"] != ""))
                else (
                    (x["Done"] - x["In Testing"])
                    if ((x["In Testing"] != "") & ((x["Done"] != "")))
                    else 0
                ),
                axis=1,
            )
            df_temp7 = df_temp7.merge(issueChangeLog, on="Issue key")
            # convert timedelta to hours since mean is not possibel on timedelta
            df_temp7["IQE Turn Around Time"] = df_temp7[
                "IQE Turn Around Time"
            ] / pd.Timedelta(hours=1)
            df_temp7 = (
                df_temp7[["Sprint id", "IQE Turn Around Time"]]
                .groupby(["Sprint id"])[["IQE Turn Around Time"]]
                .mean()
            )  # 3 days 01:53:16.528000
            # convert float back to timedelta
            df_temp7["IQE Turn Around Time"] = pd.to_timedelta(
                df_temp7["IQE Turn Around Time"], unit="hours"
            )
            op = op.merge(
                df_temp7[["IQE Turn Around Time"]], on=["Sprint id"], how="left"
            )

        df_temp8 = df_temp2[df_temp2["Issue Type.name"] == "Bug"]
        if df_temp8.shape[0] != 0:
            df_temp8 = issueStatusChange(
                jira_conn, df_temp8, [["In Progress", "In Testing"]], i
            )
        if df_temp8.shape[0] != 0:
            issueChangeLog = getChangeLogForIssue(
                jira_conn, df_temp8, [["In Progress", "In Testing"]], issueType=""
            )
            issueChangeLog["Dev Turn Around Time"] = issueChangeLog.apply(
                lambda x: (x["In Testing"] - x["In Progress"])
                if ((x["In Testing"] != "") & (x["In Progress"] != ""))
                else 0,
                axis=1,
            )
            df_temp8 = df_temp8.merge(issueChangeLog, on="Issue key")
            df_temp8["Dev Turn Around Time"] = df_temp8[
                "Dev Turn Around Time"
            ] / pd.Timedelta(hours=1)
            # df_temp8 =df_temp8.drop_duplicates(['Issue key','In Testing','In Progress','Dev Turn Around Time'])
            df_temp8 = (
                df_temp8[["Sprint id", "Dev Turn Around Time"]]
                .groupby(["Sprint id"])
                .mean()[["Dev Turn Around Time"]]
            )
            df_temp8["Dev Turn Around Time"] = pd.to_timedelta(
                df_temp8["Dev Turn Around Time"], unit="hours"
            )
            op = op.merge(
                df_temp8[["Dev Turn Around Time"]], on=["Sprint id"], how="left"
            )
        # DEV turn around time
        final_df = pd.concat([final_df, op], axis=0, ignore_index=True)

    if len(date_range) != 0:
        print("Filtering for given date range.....")
        final_df = final_df[
            (final_df["Sprint startDate"] >= date_range[0])
            & (final_df["Sprint startDate"] <= date_range[1])
        ]

    if len(sprints) != 0:
        print("Filtering for given Sprints.....")
        final_df = final_df[final_df["Sprint name"].isin(sprints)]
    columns_required = [
        "# User stories commited",
        "# User Storied Carried From Previous Sprint",
        "# User Stories Spilled To Next Sprint (IQE)",
        "# User Stories Spilled To Next Sprint (Dev)",
        "# User Stories Delivered for IQE to Test",
        "# User stories delivered on time to IQE",
        "# User Stories Completed by IQE",
        "# User Stories Certified by IQE",
        "# User Stories Rejected by IQE",
        "# User Stories Completed",
        "# User Stories Passed by IQE but Rejected by PO",
        "# of Defects Logged by IQE (In-Sprint)",
        "Defects Re-tested (In-Sprint + Regression)",
        "# Defects Rejected (In-Sprint)",
        "Defects Slipped by Dev Team",
        "# of Defects Logged from Regression Execution)",
        "# Defects Open",
        "# Defects Closed",
        "# Defects Rejected",
        "# of Defects reported in UAT",
        "Re-opened Defects Count(UAT)",
        "High Priority(UAT defects)",
        "Medium Priority(UAT defects)",
        "Low Priority(UAT defects)",
        "IQE Turn Around Time",
        "Dev Turn Around Time",
    ]

    for i in columns_required:
        if i not in final_df.columns:
            final_df[i] = np.nan

    final_df.fillna(
        {
            "IQE Turn Around Time": pd.Timedelta("0 days"),
            "Dev Turn Around Time": pd.Timedelta("0 days"),
        },
        inplace=True,
    )
    cols = list(final_df.columns)
    cols.remove("IQE Turn Around Time")
    cols.remove("Dev Turn Around Time")

    # convert timedelta64 columns to string type
    for i in ["IQE Turn Around Time", "Dev Turn Around Time"]:
        final_df[i] = final_df[i].astype("str")
    for i in cols:
        final_df.fillna({i: 0}, inplace=True)
    now = datetime.now()
    ts_now = str(now.strftime("%m-%d-%Y_%H.%M.%S"))
    fname = fname + "_" + ts_now + ".xlsx"
    writer = pd.ExcelWriter(fname, engine="xlsxwriter")
    print("Writing issues to file " + fname + ".....")
    final_df[["Project.name", "Sprint name"] + columns_required].to_excel(
        writer, sheet_name="Sprintinfo", index=False, encoding="utf-8"
    )
    print("Saving file " + fname + ".....")
    writer.save()


if __name__ == "__main__":
    create_reports()
