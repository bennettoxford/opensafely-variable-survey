# Creating job server data dump

1. [Restore](https://github.com/opensafely-core/job-server/blob/main/DEVELOPERS.md#restoring-backups) a new copy of the job-server database dump
2. Within the job-server project diretory, open a database shell with `just manage dbshell`
3. Run the following SQL command

```sql

\copy (select r.url, jr.sha, jr.created_at, jr._status from jobserver_workspace w join jobserver_repo r on w.repo_id = r.id join jobserver_jobrequest jr on jr.workspace_id = w.id) to 'all_jobs.csv' with csv header;
```
