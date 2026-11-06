"""Agent checks without model calls: tool schemas and the 20/10 image context policy."""
from types import SimpleNamespace
import unittest

from google.genai import types
from cua_embodied import GeminiAgent, Media, Observation, Task
from cua_embodied.agent import AgentAction, EARLIER_IMAGE_PLACEHOLDER
from cua_embodied.computer import PixelComputer


class AgentTests(unittest.TestCase):
    def test_schemas_still_expose_only_mouse_keyboard(self):
        self.assertEqual([t.name for t in PixelComputer.tool_definitions], ["mouse", "keyboard"])
        for tool in PixelComputer.tool_definitions:
            self.assertIn("sequence", tool.parameters["properties"])
            self.assertIn("repeat", tool.parameters["properties"])
            self.assertFalse(tool.parameters["additionalProperties"])
            types.FunctionDeclaration(name=tool.name, parameters_json_schema=tool.parameters)

    def test_mouse_schema_matches_normalized_driver_contract(self):
        schema = PixelComputer.tool_definitions[0].parameters["properties"]
        for fields in (schema, schema["sequence"]["items"]["properties"]):
            for axis in ("x", "y", "to_x", "to_y"):
                self.assertEqual(fields[axis]["minimum"], 0)
                self.assertEqual(fields[axis]["maximum"], 1000)

    def test_twenty_images_drop_oldest_ten_preserving_text_video(self):
        requests = []
        def generate(**kwargs):
            requests.append(kwargs)
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
                role="model", parts=[types.Part(function_call=types.FunctionCall(
                    name="keyboard", args={"keys": ["w"], "repeat": 2}))]))])
        agent = GeminiAgent(client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                            image_context_mode="sliding", image_context_window=20, image_context_drop=10)
        agent.register_tools(PixelComputer.tool_definitions)
        agent.begin_task(Task("fixture", "Navigate."), software_description=PixelComputer.description,
                         observation=Observation(text="Initial text", media=(Media(mime_type="image/png", data=b"0"),)))
        # A pre-uploaded video part isolates folding; this test performs no upload.
        agent._contents[0].parts.append(types.Part.from_uri(file_uri="https://example.invalid/video", mime_type="video/mp4"))
        for index in range(1, 20):
            agent.observe(AgentAction("keyboard", {"keys": ["w"]}), {"ok": True},
                          Observation(text=f"Observation {index}", media=(Media(mime_type="image/png", data=str(index).encode()),)))
        action = agent.act()
        self.assertEqual(action.arguments["repeat"], 2)
        parts = [p for c in requests[0]["contents"] for p in c.parts]
        images = [p.inline_data.data for p in parts if p.inline_data and p.inline_data.mime_type.startswith("image/")]
        self.assertEqual(images, [str(i).encode() for i in range(10, 20)])
        self.assertEqual(sum(p.text == EARLIER_IMAGE_PLACEHOLDER for p in parts), 10)
        self.assertEqual(sum(bool(p.file_data and p.file_data.mime_type == "video/mp4") for p in parts), 1)
        self.assertTrue(any(p.text == "Initial text" for p in parts))
        self.assertTrue(any(p.text == "Observation 19" for p in parts))



class DefaultsTests(unittest.TestCase):
    def test_default_context_policy_matches_paper(self):
        agent = GeminiAgent(client=SimpleNamespace())
        self.assertEqual((agent.image_context_mode, agent.image_context_window,
                          agent.image_context_drop), ("sliding", 20, 10))


if __name__ == "__main__":
    unittest.main()
