import dataclasses
import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape
from langchain.agents import AgentState
from config.configuration import Configuration
# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

#加载prompt模版
def get_prompt_template(prompt_name: str, locale: str = "en-US") ->str:
    try:
        # Normalize locale format
        normalized_locale = locale.replace("-", "_") if locale and locale.strip() else "en_US"

        try:
            template = env.get_template(f"{prompt_name}.{normalized_locale}.md")
            return template.render()
        except TemplateNotFound:
            template = env.get_template(f"{prompt_name}.md")
            return template.reader()
    except Exception as e:
        raise ValueError(f"Error loading template... : {e}")
    
#生成message列表
def apply_prompt_template(prompt_name: str, state: AgentState, configurable: Configuration = None, locale: str = "en-US") -> list:

    try:
        system_prompt = get_system_prompt_template(prompt_name, state, configurable, locale)
        return [{"role":"system","content": system_prompt}] + state["messages"]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name} for locale {locale}: {e}")

#构造system prompt
def get_system_prompt_template(prompt_name: str, state: AgentState, configurable: Configuration = None, locale: str = "en-US") ->str:
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }

    # Add configurable variables
    if configurable:
        state_vars.update(dataclasses.asdict(configurable))

    try:
        # Normalize locale format
        normalized_locale = locale.replace("-", "_") if locale and locale.strip() else "en_US"

        # Try locale-specific template first
        try:
            template = env.get_template(f"{prompt_name}.{normalized_locale}.md")
        except TemplateNotFound:
            # Fallback to English template
            template = env.get_template(f"{prompt_name}.md")

        system_prompt = template.render(**state_vars)
        return system_prompt
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name} for locale {locale}: {e}")
