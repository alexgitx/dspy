import os
import json
from datetime import datetime
from smartflow_dspy_conversational_agent import EventAnalysisPipeline
from typing import Dict


class MemoryStore:
    def __init__(self, client_id: str, base_path="memories"):
        self.client_id = client_id
        self.path = f"{base_path}/{client_id}.json"
        os.makedirs(base_path, exist_ok=True)
        self.history = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_event(self, event_data: Dict):
        self.history.append(event_data)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def get_context_summary(self, max_events=5) -> str:
        recent = self.history[-max_events:]
        return "\n".join([f"{e['timestamp']}: {e['event']}" for e in recent])


class ClientEventPipeline:
    def __init__(self, client_id: str):
        self.memory = MemoryStore(client_id)
        self.pipeline = EventAnalysisPipeline()

    def analyze(self, event_text: str) -> Dict:
        context_summary = self.memory.get_context_summary()
        enriched_event = (
            f"{context_summary}\n\nNuevo evento: {event_text}"
            if context_summary
            else event_text
        )
        result = self.pipeline.analyze(enriched_event)
        if result["success"]:
            self.memory.save_event(
                {
                    "timestamp": str(datetime.utcnow()),
                    "event": event_text,
                    "recommendation": result["recommendation"],
                }
            )
        return result


# Ejemplo de uso
if __name__ == "__main__":
    client_id = "cliente_abc"
    event = "El cliente ha abierto 3 tickets críticos en 24 horas."
    agent = ClientEventPipeline(client_id)
    output = agent.analyze(event)
    print(json.dumps(output, indent=2, ensure_ascii=False))
