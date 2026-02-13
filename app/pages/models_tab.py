"""Models tab - View and reload models"""
import gradio as gr

from app import config
from app.models import get_models_list, reload_all_models


def show_models():
    """Display current models"""
    info = get_models_list()
    return f"**Total Models:** {info['count']}\n\n**Models:**\n" + "\n".join(f"- {m}" for m in info['models'])


def reload_models_handler():
    """Reload all models from HuggingFace and local directory"""
    old_count, new_count, available_models = reload_all_models()
    
    info = get_models_list()
    status = f"🔄 **Reloaded!** {old_count} → {new_count} models\n\n"
    status += f"**Total Models:** {info['count']}\n\n**Models:**\n"
    status += "\n".join(f"- {m}" for m in info['models'])
    
    return status, gr.update(choices=available_models), gr.update(choices=available_models), gr.update(choices=available_models), gr.update(choices=available_models)


def create_models_tab(model_dropdown, p1_model, p2_model, baseline_model_dropdown):
    """Create the Models tab UI"""
    gr.Markdown("### Available Models")
    
    models_output = gr.Markdown(show_models())
    refresh_btn = gr.Button("🔄 Reload Models from HF", variant="primary")
    refresh_btn.click(
        fn=reload_models_handler, 
        outputs=[models_output, model_dropdown, p1_model, p2_model, baseline_model_dropdown]
    )
