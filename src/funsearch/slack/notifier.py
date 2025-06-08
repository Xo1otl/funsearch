import os
import json
import urllib.request
from funsearch import presenter

try:
    webhook_url = os.environ["SLACK_WEBHOOK_URL"]
except KeyError:
    from .env import WEBHOOK_URL
    webhook_url = WEBHOOK_URL


class SlackNotifier(presenter.ResultNotifier):
    """Slack通知を送信するクラス"""
    
    def send_message(self, message: str) -> bool:
        """
        Webhook経由でSlackにメッセージを送信
        
        Args:
            message: 送信するメッセージ
            
        Returns:
            送信成功時True、失敗時False
        """
        payload = {"text": message}
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"Slack webhook error: {e}")
            return False