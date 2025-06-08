from jinja2 import Template

REASONING_PROMPT_TEMPLATE = Template("""
You are an expert at analyzing GUI automation actions. Given a goal, an action taken by a GUI agent, and screenshots before and after the action, provide clear reasoning for taking this action as if you were the agent.

## Goal
{{ goal }}

{% if action_history -%}
## Action History
{%- for action in action_history %}
{{ loop.index }}. {{ action }}
{% endfor -%}
{%- endif %}

## Current Action
{{ action }}

## Instructions
1. Analyze the action history and the before and after screenshots to understand what changed
2. Provide the reasoning for taking this action if you were the agent trying to explain the action to a human. Write in future tense.
3. Be concise but thorough (2-3 sentences). Refer to the specific UI elements instead of vague terms like "click" or "select" certain coordinates.

## Examples

### Example 1

Goal: I want to show bookmarks bar by default in chrome.
Action: Click the chrome button in desktop
Reasoning: To enable the bookmarks bar by default in Chrome, I first need to
open the browser. The next logical step is to double-click on the Chrome icon on the desktop to launch the application and access its settings. Double-click on the Chrome icon on the desktop to open the browser.

## Your Response
Please provide only the reasoning text, no additional formatting or explanations.
Reasoning: """)
