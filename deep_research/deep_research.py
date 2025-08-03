import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager
import asyncio
import pyperclip

load_dotenv(override=True)

manager = ResearchManager()

async def run(query: str, set_status):
    async for chunk in manager.run(query):
        set_status(chunk)
    return chunk  # Final report

def show_page(page: str):
    return gr.update(visible=(page == "welcome")), gr.update(visible=(page == "research")), gr.update(visible=(page == "progress")), gr.update(visible=(page == "result"))

with gr.Blocks(theme=gr.themes.Ocean(primary_hue="sky")) as ui:
    page_state = gr.State("welcome")
    final_report = gr.State("")

    # Clarifying question and answer states
    clar_question1 = gr.State("")
    clar_question2 = gr.State("")
    clar_question3 = gr.State("")

    # --- Welcome Page ---
    with gr.Column(visible=True) as welcome_page:
        gr.Markdown("""
        <div style='text-align:center'>
            <h1>🤖 Your Trading Assistant</h1>
            <h3>Welcome! Click the button below to start your smart market research</h3>
        </div>
        """)
        start_btn = gr.Button("🚀 Start Research", elem_id="start-btn", variant="primary")

    # --- Research Page ---
    with gr.Column(visible=False) as research_page:
        gr.Markdown("### What would you like to research today?")
        query_textbox = gr.Textbox(label="Enter your question or research topic")
        run_button = gr.Button("🔍 Start Search", variant="primary")
        status = gr.Markdown("Waiting for your question...", elem_id="status")
        back_btn = gr.Button("⬅️ Back", variant="secondary")


    with gr.Column(visible=False) as progress_page:
        progress_status = gr.Markdown("<div style='text-align:center; font-size:2em; margin-top:2em;'>Initializing...</div>")
        progress_anim = gr.HTML("""
        <div style='display:flex; justify-content:center; margin-top:1em;'>
            <svg width="60" height="60" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" stroke="#00bfff">
                <g fill="none" fill-rule="evenodd" stroke-width="4">
                    <circle cx="22" cy="22" r="18" stroke-opacity=".5"/>
                    <path d="M40 22c0-9.94-8.06-18-18-18">
                        <animateTransform attributeName="transform" type="rotate" from="0 22 22" to="360 22 22" dur="1s" repeatCount="indefinite"/>
                    </path>
                </g>
            </svg>
        </div>
        """)

    # --- Result Page ---
    with gr.Column(visible=False) as result_page:
        gr.Markdown("### Research Report Ready!")
        report_md = gr.Markdown()
        copy_btn = gr.Button("📋 Copy Report")
        new_search_btn = gr.Button("🔄 New Research")
        gr.Markdown("<div style='text-align:center; color:#00cc00'>Thank you for using the Trading Assistant!</div>")


    # --- Clarification Page 1 ---
    with gr.Column(visible=False) as clarification_page1:
        clar_q1_text = gr.Textbox(label="Clarifying Question 1", interactive=False)
        clar_answer1 = gr.Textbox(label="Your answer to Question 1")
        next_q1_btn = gr.Button("Next")


    # --- Clarification Page 2 ---
    with gr.Column(visible=False) as clarification_page2:
        clar_q2_text = gr.Textbox(label="Clarifying Question 2", interactive=False)
        clar_answer2 = gr.Textbox(label="Your answer to Question 2")
        next_q2_btn = gr.Button("Next")


    # --- Clarification Page 3 ---
    with gr.Column(visible=False) as clarification_page3:
        clar_q3_text = gr.Textbox(label="Clarifying Question 3", interactive=False)
        clar_answer3 = gr.Textbox(label="Your answer to Question 3")
        submit_answers_btn = gr.Button("Submit Answers")


    # --- Page navigation logic ---
    def go_to_research():
        return (
            "research",  # page_state
            *show_page("research"),
            "Waiting for your question...",  # status
            ""  # report_md
        )
    def go_to_result(report):
        return (
            "result",
            report,
            *show_page("result"),
            "Waiting for your question...",  # status (reset for next time)
            report  # report_md (reset for next time)
        )
    def go_to_welcome():
        return (
            "welcome",         # page_state
            "",                # final_report
            *show_page("welcome"),
            "Waiting for your question...",  # status
            ""                 # report_md
        )
    def go_to_progress():
        return (
            "progress",  # page_state
            *show_page("progress"),
            "<div style='text-align:center; font-size:2em; margin-top:2em;'>Starting research...</div>",  # progress_status
            ""  # report_md (not used here, but keep output count consistent)
        )
    def show_clarification_page(page: int):
        return (
            gr.update(visible=(page == 1)),  # clarification_page1
            gr.update(visible=(page == 2)),  # clarification_page2
            gr.update(visible=(page == 3)),  # clarification_page3
        )

    start_btn.click(
        go_to_research,
        outputs=[page_state, welcome_page, research_page, progress_page, result_page, status, report_md]
    )
    back_btn.click(
        go_to_welcome,
        outputs=[page_state, final_report, welcome_page, research_page, progress_page, result_page, status, report_md]
    )
    new_search_btn.click(
        go_to_research,
        outputs=[page_state, welcome_page, research_page, progress_page, result_page, status, report_md]
    )
    next_q1_btn.click(
        lambda: show_clarification_page(2),
        outputs=[clarification_page1, clarification_page2, clarification_page3]
    )

    next_q2_btn.click(
        lambda: show_clarification_page(3),
        outputs=[clarification_page1, clarification_page2, clarification_page3]
    )


    # --- Research execution ---
    async def do_research(query):
        report = ""
        async for chunk in manager.run(query):
            if chunk.strip().startswith("#") or len(chunk) > 500:
                report = chunk
            else:
                # Show status big and centered
                yield f"<div style='text-align:center; font-size:2em; margin-top:2em;'>{chunk}</div>", ""
        # After research is done, hide progress and show result
        yield "", report


    async def analyze_and_maybe_clarify(query):
        global clar_q1_text, clar_q2_text, clar_q3_text
        clarifying_data = await manager.question_analysis(query)

        if clarifying_data.clarify:
            # Debug print statements OUTSIDE the return
            print("clar_q1_text:", clar_q1_text)
            print("clar_q2_text:", clar_q2_text)
            print("clar_q3_text:", clar_q3_text)

            return (
                "clarify",
                clarifying_data.question1,
                clarifying_data.question2,
                clarifying_data.question3,
                *show_clarification_page(1),
                gr.update(value=clarifying_data.question1),
                gr.update(value=clarifying_data.question2),
                gr.update(value=clarifying_data.question3),            
            )
        else:
            return await do_research(query)
    
    
    run_button.click(
        analyze_and_maybe_clarify,
        inputs=[query_textbox],
        outputs=[
            page_state,
            clar_question1, clar_question2, clar_question3,
            clarification_page1, clarification_page2, clarification_page3,
            clar_q1_text, clar_q2_text, clar_q3_text
        ]
    )

    async def continue_research_after_clarification(query, a1, a2, a3, q1, q2, q3):
        answers = [a1, a2, a3]
        questions = [q1, q2, q3]

        from clarifier_agent import ClarifyingQuestions
        analysis = ClarifyingQuestions(
            clarify=True,
            question1=q1,
            question2=q2,
            question3=q3
        )

        report = ""

        try:
            search_plan = await manager.plan_searches(query, analysis, answers)
            search_results = await manager.perform_searches(search_plan)
            report_data = await manager.write_report(query, search_results)
            await manager.send_email(report_data)
            report = report_data.markdown_report
        except Exception as e:
            report = f"An error occurred: {e}"

        return (
            "result",
            report,
            *show_page("result"),
            "Waiting for your question...",
            report
        )


    submit_answers_btn.click(
    continue_research_after_clarification,
    inputs=[
        query_textbox,
        clar_answer1, clar_answer2, clar_answer3,
        clar_question1, clar_question2, clar_question3
    ],
    outputs=[
        page_state, final_report,
        welcome_page, research_page, progress_page, result_page,
        status, report_md
    ]
)


    # --- Copy to clipboard ---
    def copy_report(report):
        pyperclip.copy(report)
        return gr.update(value="📋 Copied!")

    copy_btn.click(copy_report, inputs=[final_report], outputs=[copy_btn])

ui.launch(inbrowser=True)

"""
import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager

load_dotenv(override=True)


async def run(query: str):
    async for chunk in ResearchManager().run(query):
        yield chunk


with gr.Blocks(theme=gr.themes.Ocean(primary_hue="sky")) as ui:
    gr.Markdown("# Trading Assistant")
    query_textbox = gr.Textbox(label="What are we looking for today?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

"""