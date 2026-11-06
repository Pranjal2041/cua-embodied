"""Task instructions for OpenSUN3D, shown in the GUI and given to the agent."""

SCENEFUN3D_TASK_RULES = """Your output is an affordance/contact-component mask, not automatically the whole object named in the objective. Select the smallest complete, coherent physical component that a person, plug, or tool would directly touch or engage to perform the requested action. Exclude the larger parent object's body or surface whenever a smaller actionable component exists. Do not shrink the answer to a microscopic contact point: select the complete actionable component.

Apply these concrete rules:
- Open or close a drawer/chest: select its handle or pull, never the drawer front or body.
- Open or close a closet/cabinet/room door: select its handle or pull, never the door panel.
- Plug a device in: select the one specific empty socket/receptacle that should receive the plug, never the inserted plug, complete multi-socket plate, power strip, or surrounding wall.
- Unplug a lamp/device: select the inserted plug head, never the socket plate, cable, or lamp.
- Turn on/off a light or appliance: select its switch or button, never the whole fixture.
- Adjust a device: select its dial, knob, slider, or button, never the appliance body.

Before every click, determine internally: (1) the parent object, (2) the actionable component, and (3) what directly makes physical contact. Click only that component. Segmentation scores measure confidence, not semantic correctness: reject a high-score candidate if it mostly covers the parent object or surrounding surface. If every candidate is broad or ambiguous, move closer, change viewpoint, and click again. Call exactly one tool per turn. Do not submit until you have inspected and accepted an actual, precise target mask. Prefer a second meaningful viewpoint when it can disambiguate the component."""
