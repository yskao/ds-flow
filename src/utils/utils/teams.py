"""Functions of teams operation."""
from prefect.blocks.notifications import MicrosoftTeamsWebhook


def send_message_to_teams(channel: str, message: str) -> None:
    """Send message to teams."""
    teams_webhook_block = MicrosoftTeamsWebhook.load(channel)
    teams_webhook_block.notify(message)
