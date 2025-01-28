from langgraph_sdk import Auth
import logging

logger = logging.getLogger(__name__)

auth = Auth()


# Very permissive.
@auth.authenticate
def authenticate():
    return "Authenticated"


@auth.on
async def block(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    logger.warning(f"Accepting (should be blocking) {ctx.user.identity} with {ctx.resource} / {ctx.action}")
    # assert False
    return None


@auth.on.threads
async def accept(ctx: Auth.types.AuthContext, value: Auth.types.on.threads.value):
    logger.warning(f"Accepting {ctx.user.identity} with {ctx.resource} / {ctx.action}.")
    return None
