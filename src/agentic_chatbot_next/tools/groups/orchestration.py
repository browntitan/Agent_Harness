from __future__ import annotations

import json
from typing import Any, List

from langchain_core.tools import tool


def build_orchestration_tools(ctx: Any) -> List[Any]:
    if ctx.kernel is None or ctx.active_definition is None:
        return []

    @tool
    def spawn_worker(
        prompt: str,
        agent_name: str = "utility",
        description: str = "",
        run_in_background: bool = False,
    ) -> str:
        """Spawn a scoped worker from the current next runtime."""

        return json.dumps(
            ctx.kernel.spawn_worker_from_tool(
                ctx,
                prompt=prompt,
                agent_name=agent_name,
                description=description,
                run_in_background=run_in_background,
            ),
            ensure_ascii=False,
        )

    @tool
    def message_worker(job_id: str, message: str, resume: bool = True) -> str:
        """Queue a follow-up message for an existing worker job."""

        return json.dumps(
            ctx.kernel.message_worker_from_tool(
                ctx,
                job_id=job_id,
                message=message,
                resume=resume,
            ),
            ensure_ascii=False,
        )

    @tool(return_direct=True)
    def request_parent_question(
        question: str,
        reason: str = "",
        options: List[str] | None = None,
        context: str = "",
    ) -> str:
        """Pause this worker and ask the parent/coordinator for an answer."""

        return json.dumps(
            ctx.kernel.request_parent_question_from_tool(
                ctx,
                question=question,
                reason=reason,
                options=options or [],
                context=context,
            ),
            ensure_ascii=False,
        )

    @tool(return_direct=True)
    def request_parent_approval(
        action: str,
        reason: str,
        tool_name: str = "",
        arguments: dict | None = None,
        risk: str = "",
        context: str = "",
    ) -> str:
        """Pause this worker and ask an operator to approve or deny an action."""

        return json.dumps(
            ctx.kernel.request_parent_approval_from_tool(
                ctx,
                action=action,
                reason=reason,
                tool_name=tool_name,
                arguments=arguments or {},
                risk=risk,
                context=context,
            ),
            ensure_ascii=False,
        )

    @tool
    def list_worker_requests(
        job_id: str = "",
        status_filter: str = "open",
        request_type: str = "",
    ) -> str:
        """List pending typed requests from worker jobs."""

        return json.dumps(
            ctx.kernel.list_worker_requests_from_tool(
                ctx,
                job_id=job_id,
                status_filter=status_filter,
                request_type=request_type,
            ),
            ensure_ascii=False,
        )

    @tool
    def respond_worker_request(
        job_id: str,
        request_id: str,
        response: str,
        decision: str = "",
        resume: bool = True,
    ) -> str:
        """Answer a worker question request. Approval requests require operator API approval."""

        return json.dumps(
            ctx.kernel.respond_worker_request_from_tool(
                ctx,
                job_id=job_id,
                request_id=request_id,
                response=response,
                decision=decision,
                resume=resume,
            ),
            ensure_ascii=False,
        )

    @tool
    def invoke_agent(
        agent_name: str,
        message: str,
        description: str = "",
        job_id: str = "",
        reuse_running_job: bool = True,
        team_channel_id: str = "",
    ) -> str:
        """Queue an async peer request for another allowed agent in this session."""

        return json.dumps(
            ctx.kernel.invoke_agent_from_tool(
                ctx,
                agent_name=agent_name,
                message=message,
                description=description,
                job_id=job_id,
                reuse_running_job=reuse_running_job,
                team_channel_id=team_channel_id,
            ),
            ensure_ascii=False,
        )

    @tool
    def create_team_channel(
        name: str,
        purpose: str = "",
        member_agents: List[str] | None = None,
        member_job_ids: List[str] | None = None,
    ) -> str:
        """Create a same-session team mailbox channel."""

        return json.dumps(
            ctx.kernel.create_team_channel_from_tool(
                ctx,
                name=name,
                purpose=purpose,
                member_agents=member_agents or [],
                member_job_ids=member_job_ids or [],
            ),
            ensure_ascii=False,
        )

    @tool
    def post_team_message(
        channel_id: str,
        content: str,
        message_type: str = "message",
        target_agents: List[str] | None = None,
        target_job_ids: List[str] | None = None,
        subject: str = "",
        payload: dict | None = None,
    ) -> str:
        """Post a typed message into a team mailbox channel."""

        return json.dumps(
            ctx.kernel.post_team_message_from_tool(
                ctx,
                channel_id=channel_id,
                content=content,
                message_type=message_type,
                target_agents=target_agents or [],
                target_job_ids=target_job_ids or [],
                subject=subject,
                payload=payload or {},
            ),
            ensure_ascii=False,
        )

    @tool
    def list_team_messages(
        channel_id: str = "",
        message_type: str = "",
        status_filter: str = "open",
        limit: int = 20,
    ) -> str:
        """List team mailbox messages visible to the current agent."""

        return json.dumps(
            ctx.kernel.list_team_messages_from_tool(
                ctx,
                channel_id=channel_id,
                message_type=message_type,
                status_filter=status_filter,
                limit=limit,
            ),
            ensure_ascii=False,
        )

    @tool
    def claim_team_messages(
        channel_id: str,
        limit: int = 0,
        message_type: str = "",
    ) -> str:
        """Claim open team mailbox messages for this agent/job."""

        return json.dumps(
            ctx.kernel.claim_team_messages_from_tool(
                ctx,
                channel_id=channel_id,
                limit=limit,
                message_type=message_type,
            ),
            ensure_ascii=False,
        )

    @tool
    def respond_team_message(
        channel_id: str,
        message_id: str,
        response: str,
        decision: str = "",
        resolve: bool = True,
    ) -> str:
        """Answer a team mailbox question request. Approvals require operator API approval."""

        return json.dumps(
            ctx.kernel.respond_team_message_from_tool(
                ctx,
                channel_id=channel_id,
                message_id=message_id,
                response=response,
                decision=decision,
                resolve=resolve,
            ),
            ensure_ascii=False,
        )

    @tool
    def list_jobs(status_filter: str = "") -> str:
        """List durable runtime jobs for the current session."""

        return json.dumps(
            ctx.kernel.list_jobs_from_tool(
                ctx,
                status_filter=status_filter,
            ),
            ensure_ascii=False,
        )

    @tool
    def stop_job(job_id: str) -> str:
        """Stop a background worker job."""

        return json.dumps(
            ctx.kernel.stop_job_from_tool(
                ctx,
                job_id=job_id,
            ),
            ensure_ascii=False,
        )

    return [
        spawn_worker,
        message_worker,
        request_parent_question,
        request_parent_approval,
        list_worker_requests,
        respond_worker_request,
        invoke_agent,
        create_team_channel,
        post_team_message,
        list_team_messages,
        claim_team_messages,
        respond_team_message,
        list_jobs,
        stop_job,
    ]
