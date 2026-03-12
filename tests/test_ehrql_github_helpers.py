from datetime import UTC, datetime, timedelta

from parsing import ehrql_github_helpers


def _graphql_repo_page(nodes: list[dict], has_next_page: bool = False) -> dict:
    return {
        "data": {
            "organization": {
                "repositories": {
                    "nodes": nodes,
                    "pageInfo": {
                        "hasNextPage": has_next_page,
                        "endCursor": "cursor-1" if has_next_page else None,
                    },
                }
            }
        }
    }


def test_bootstrap_repo_head_sync_runs_full_graphql_query(monkeypatch):
    now = datetime(2026, 3, 12, 12, 0, tzinfo=UTC)
    calls = []
    saved_cache = {}

    monkeypatch.setattr(ehrql_github_helpers, "_load_repo_heads_cache", lambda: {})
    monkeypatch.setattr(ehrql_github_helpers, "_utc_now", lambda: now)
    monkeypatch.setattr(
        ehrql_github_helpers,
        "_save_repo_heads_cache",
        lambda cache_data: saved_cache.update(cache_data),
    )

    def fake_run_gh(args: list[str], expect_json: bool = True) -> dict:
        calls.append(args)
        return _graphql_repo_page(
            [
                {
                    "name": "repo-a",
                    "pushedAt": "2026-03-11T09:00:00Z",
                    "defaultBranchRef": {"target": {"oid": "sha-new"}},
                }
            ]
        )

    monkeypatch.setattr(ehrql_github_helpers, "run_gh", fake_run_gh)

    result = ehrql_github_helpers.get_all_repos_with_shas_graphql()

    assert result == {"repo-a": "sha-new"}
    assert len(calls) == 1
    assert any(arg == "first=100" for arg in calls[0])
    assert saved_cache["last_checked_at"] == ehrql_github_helpers._to_iso8601_utc(now)
    assert saved_cache["last_full_sync_at"] == ehrql_github_helpers._to_iso8601_utc(now)


def test_existing_cache_uses_incremental_sync_even_if_last_full_sync_is_old(
    monkeypatch,
):
    now = datetime(2026, 3, 12, 12, 0, tzinfo=UTC)
    last_checked_at = now - timedelta(days=21)
    calls = []
    saved_cache = {}
    cache_data = {
        "last_checked_at": ehrql_github_helpers._to_iso8601_utc(last_checked_at),
        "last_full_sync_at": ehrql_github_helpers._to_iso8601_utc(
            now - timedelta(days=30)
        ),
        "repos": {
            "repo-a": {
                "sha": "sha-old",
                "pushed_at": "2026-02-15T09:00:00Z",
            }
        },
    }

    monkeypatch.setattr(
        ehrql_github_helpers, "_load_repo_heads_cache", lambda: dict(cache_data)
    )
    monkeypatch.setattr(ehrql_github_helpers, "_utc_now", lambda: now)
    monkeypatch.setattr(
        ehrql_github_helpers,
        "_save_repo_heads_cache",
        lambda updated_cache: saved_cache.update(updated_cache),
    )

    def fake_run_gh(args: list[str], expect_json: bool = True) -> dict:
        calls.append(args)
        return _graphql_repo_page(
            [
                {
                    "name": "repo-a",
                    "pushedAt": "2026-02-18T00:00:00Z",
                    "defaultBranchRef": {"target": {"oid": "sha-fresh"}},
                }
            ],
            has_next_page=True,
        )

    monkeypatch.setattr(ehrql_github_helpers, "run_gh", fake_run_gh)

    result = ehrql_github_helpers.get_all_repos_with_shas_graphql()

    assert result["repo-a"] == "sha-fresh"
    assert len(calls) == 1
    assert any(arg == "first=20" for arg in calls[0])
    assert not any(arg == "first=100" for arg in calls[0])
    assert saved_cache["last_checked_at"] == ehrql_github_helpers._to_iso8601_utc(now)
    assert saved_cache["last_full_sync_at"] == cache_data["last_full_sync_at"]


def test_existing_cache_without_last_checked_falls_back_to_incremental_from_full_sync(
    monkeypatch,
):
    now = datetime(2026, 3, 12, 12, 0, tzinfo=UTC)
    last_full_sync_at = now - timedelta(days=14)
    calls = []
    cache_data = {
        "last_full_sync_at": ehrql_github_helpers._to_iso8601_utc(last_full_sync_at),
        "repos": {
            "repo-a": {
                "sha": "sha-old",
                "pushed_at": "2026-02-20T09:00:00Z",
            }
        },
    }

    monkeypatch.setattr(
        ehrql_github_helpers, "_load_repo_heads_cache", lambda: dict(cache_data)
    )
    monkeypatch.setattr(ehrql_github_helpers, "_utc_now", lambda: now)
    monkeypatch.setattr(
        ehrql_github_helpers, "_save_repo_heads_cache", lambda cache_data: None
    )

    def fake_run_gh(args: list[str], expect_json: bool = True) -> dict:
        calls.append(args)
        return _graphql_repo_page(
            [
                {
                    "name": "repo-a",
                    "pushedAt": "2026-02-25T00:00:00Z",
                    "defaultBranchRef": {"target": {"oid": "sha-fresh"}},
                }
            ],
            has_next_page=True,
        )

    monkeypatch.setattr(ehrql_github_helpers, "run_gh", fake_run_gh)

    result = ehrql_github_helpers.get_all_repos_with_shas_graphql()

    assert result["repo-a"] == "sha-fresh"
    assert len(calls) == 1
    assert any(arg == "first=20" for arg in calls[0])
    assert not any(arg == "first=100" for arg in calls[0])


def test_force_full_sync_uses_full_graphql_query_even_with_valid_cache(monkeypatch):
    now = datetime(2026, 3, 12, 12, 0, tzinfo=UTC)
    calls = []
    saved_cache = {}
    cache_data = {
        "last_checked_at": ehrql_github_helpers._to_iso8601_utc(
            now - timedelta(days=7)
        ),
        "last_full_sync_at": ehrql_github_helpers._to_iso8601_utc(
            now - timedelta(days=30)
        ),
        "repos": {
            "repo-a": {
                "sha": "sha-old",
                "pushed_at": "2026-03-01T09:00:00Z",
            }
        },
    }

    monkeypatch.setattr(
        ehrql_github_helpers, "_load_repo_heads_cache", lambda: dict(cache_data)
    )
    monkeypatch.setattr(ehrql_github_helpers, "_utc_now", lambda: now)
    monkeypatch.setattr(
        ehrql_github_helpers,
        "_save_repo_heads_cache",
        lambda updated_cache: saved_cache.update(updated_cache),
    )

    def fake_run_gh(args: list[str], expect_json: bool = True) -> dict:
        calls.append(args)
        return _graphql_repo_page(
            [
                {
                    "name": "repo-a",
                    "pushedAt": "2026-03-11T09:00:00Z",
                    "defaultBranchRef": {"target": {"oid": "sha-new"}},
                }
            ]
        )

    monkeypatch.setattr(ehrql_github_helpers, "run_gh", fake_run_gh)

    result = ehrql_github_helpers.get_all_repos_with_shas_graphql(force_full_sync=True)

    assert result == {"repo-a": "sha-new"}
    assert len(calls) == 1
    assert any(arg == "first=100" for arg in calls[0])
    assert not any(arg == "first=20" for arg in calls[0])
    assert saved_cache["last_checked_at"] == ehrql_github_helpers._to_iso8601_utc(now)
    assert saved_cache["last_full_sync_at"] == ehrql_github_helpers._to_iso8601_utc(now)
