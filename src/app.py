import gradio as gr
from gradio_pdf import PDF

from src.document_store import EnhancedDocumentStore

# Global store instance
doc_store = EnhancedDocumentStore()


def process_pdf_handler(pdf_file, progress=gr.Progress()):
    if pdf_file is None:
        return "Please upload a PDF file", "", gr.update(choices=["All"], value="All")

    progress(0.1, desc="Opening PDF")

    filename = (
        pdf_file.split("/")[-1]
        if isinstance(pdf_file, str)
        else getattr(pdf_file, "name", "uploaded.pdf")
    )

    success, stats = doc_store.process_pdf(
        pdf_file,
        filename=filename,
        use_llama_index=True
    )

    progress(0.7, desc="Processing complete, building UI")

    if success:
        status_msg = f"""
**Successfully Processed:**
- File: {stats['filename']}
- Pages: {stats['total_pages']}
- Documents Found: {stats['documents_found']}
- Chunks Created: {stats['total_chunks']}
- Types: {', '.join(stats['document_types'])}
- Time: {stats['processing_time']}
"""

        structure = doc_store.get_document_structure() if hasattr(doc_store, "get_document_structure") else []
        structure_display = "\n".join([
            f"- **{doc['type']}** (Pages {doc['pages']}): {doc['chunks']} chunks"
            for doc in structure
        ]) if structure else "No document structure available."

        doc_types = ["All"] + stats["document_types"]

        return status_msg, structure_display, gr.update(choices=doc_types, value="All")

    return f"Error: {stats.get('error', 'Unknown error')}", "", gr.update(choices=["All"], value="All")


def chat_handler(message, history, doc_filter, auto_route, num_chunks):
    """Handle chat interactions."""
    history = history or []

    if not message or not str(message).strip():
        return history

    if not doc_store.is_ready:
        response = "Please upload and process a PDF document first."
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

    filter_type = None if doc_filter == "All" else doc_filter

    # Preferred path: use doc_store.query() if available
    if hasattr(doc_store, "query"):
        result = doc_store.query(
            message,
            filter_type=filter_type,
            auto_route=auto_route and filter_type is None,
            k=int(num_chunks)
        )

        response = f"{result.get('answer', 'No answer generated.')}\n\n"

        if result.get("sources"):
            response += "**Sources:**\n"
            for src in result["sources"]:
                relevance = src.get("relevance", "n/a")
                label = src.get("relevance_label", "")
                raw_score = src.get("raw_score", "n/a")

                response += (
                    f"- {src.get('doc_type', 'unknown')} "
                    f"(Pages {src.get('pages', 'unknown')}) - "
                    f"Relevance: {relevance}"
                    f"{f' [{label}]' if label else ''}"
                    f" (raw: {raw_score})\n"
                )

        confidence = result.get("confidence", 0)
        confidence_label = result.get("confidence_label", "")

        response += (
            f"\n*Retrieval Strength: {confidence:.1%}"
            f"{f' [{confidence_label}]' if confidence_label else ''} | "
            f"Filter: {result.get('filter_used', 'None')}*"
        )

    # Fallback path: retriever only
    else:
        results = doc_store.retriever.retrieve(
            query=message,
            k=int(num_chunks),
            filter_doc_type=filter_type,
            auto_route=auto_route and filter_type is None
        )

        if not results:
            response = "No relevant results found."
        else:
            chunks_text = []
            for i, (chunk, score) in enumerate(results, start=1):
                chunks_text.append(
                    f"**Result {i}** (Score: {score:.4f})\n"
                    f"{chunk.text}\n\n"
                    f"*Doc Type:* {chunk.doc_type} | "
                    f"*Page:* {chunk.page} | "
                    f"*Chunk:* {chunk.chunk_id}"
                )
            response = "\n\n---\n\n".join(chunks_text)

    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]


def update_status_bar():
    """Update the status bar with current statistics."""
    if doc_store.is_ready:
        stats = doc_store.processing_stats
        return (
            f"**Status:** Ready | "
            f"**Documents:** {stats.get('documents_found', 0)} | "
            f"**Chunks:** {stats.get('total_chunks', 0)}"
        )
    return "**Status:** Ready | **Documents:** 0 | **Chunks:** 0"


def clear_all():
    """Clear everything and reset the interface."""
    global doc_store
    doc_store = EnhancedDocumentStore()
    return (
        None,  # pdf_input
        "Waiting for PDF upload...",  # status_output
        "",  # structure_output
        gr.update(choices=["All"], value="All"),  # doc_filter
        [],  # chatbot
        "",  # msg_input
        update_status_bar()  # status_bar
    )


def process_pdf_with_status(pdf_file):
    status, structure, filter_update = process_pdf_handler(pdf_file)
    status_bar_text = update_status_bar()
    return status, structure, filter_update, status_bar_text


def chat_with_status(message, history, doc_filter, auto_route, num_chunks):
    new_history = chat_handler(message, history, doc_filter, auto_route, num_chunks)
    status_bar_text = update_status_bar()
    return new_history, status_bar_text


def ask_summary(history, doc_filter, auto_route, num_chunks):
    return chat_handler(
        "Can you provide a summary of the main points in this document?",
        history,
        doc_filter,
        auto_route,
        num_chunks
    )


def ask_lot_numbers(history, doc_filter, auto_route, num_chunks):
    return chat_handler(
        "What lot numbers or batch numbers are mentioned in these documents?",
        history,
        doc_filter,
        auto_route,
        num_chunks
    )


with gr.Blocks(title="Pharmaceutical Document Q&A System") as demo:
    gr.Markdown("""
# Pharmaceutical Document Q&A System
### Intelligent Multi-Document Analysis with Advanced RAG Pipeline
Upload a pharmaceutical blob PDF (e.g. pharma-blob-sample.pdf) to identify
document types, build a searchable index, and ask questions in natural language.
""")

    with gr.Row():
        # Left side - PDF upload
        with gr.Column(scale=2):
            pdf_input = gr.File(
                label="Upload Pharmaceutical PDF",
                file_types=[".pdf"],
                type="filepath"
            )

            with gr.Row():
                process_btn = gr.Button(
                    "Process Document",
                    variant="primary",
                    size="lg",
                    scale=2
                )
                clear_all_btn = gr.Button(
                    "Clear All",
                    variant="secondary",
                    size="lg",
                    scale=1
                )

        # Middle - Document info and settings
        with gr.Column(scale=1):
            gr.Markdown("### Document Info")
            status_output = gr.Markdown(
                value="Waiting for PDF upload..."
            )

            structure_output = gr.Markdown(
                value=""
            )

            gr.Markdown("### Retrieval Settings")

            doc_filter = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Document Type Filter",
                info="Filter search to a specific pharmaceutical document type"
            )

            auto_route = gr.Checkbox(
                value=True,
                label="Auto-Route Queries",
                info="Automatically detect the most relevant document type"
            )

            # Lowered Chunking
            num_chunks = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="Chunks to Retrieve"
            )

        # Right side - Chat interface
        with gr.Column(scale=2):
            gr.Markdown("### Ask Questions")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                elem_id="chatbot",
                show_label=False
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g., What is the lot number? What sterilization method was used?",
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Row():
                clear_chat_btn = gr.Button("Clear Chat", size="sm", scale=1)
                example_btn1 = gr.Button("Summarise this document", size="sm", scale=1)
                example_btn2 = gr.Button("Find lot numbers", size="sm", scale=1)

    # Status bar at the bottom
    with gr.Row():
        status_bar = gr.Markdown(
            value="**Status:** Ready | **Documents:** 0 | **Chunks:** 0",
            elem_id="status_bar"
        )

    # Wire up all the events
    process_btn.click(
        fn=process_pdf_with_status,
        inputs=[pdf_input],
        outputs=[status_output, structure_output, doc_filter, status_bar]
    )

    clear_all_btn.click(
        fn=clear_all,
        outputs=[pdf_input, status_output, structure_output, doc_filter,
                 chatbot, msg_input, status_bar]
    )

    msg_input.submit(
        fn=chat_with_status,
        inputs=[msg_input, chatbot, doc_filter, auto_route, num_chunks],
        outputs=[chatbot, status_bar]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )

    send_btn.click(
        fn=chat_with_status,
        inputs=[msg_input, chatbot, doc_filter, auto_route, num_chunks],
        outputs=[chatbot, status_bar]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )

    clear_chat_btn.click(
        lambda: [],
        outputs=[chatbot]
    )

    example_btn1.click(
        fn=ask_summary,
        inputs=[chatbot, doc_filter, auto_route, num_chunks],
        outputs=[chatbot]
    ).then(
        fn=update_status_bar,
        outputs=[status_bar]
    )

    example_btn2.click(
        fn=ask_lot_numbers,
        inputs=[chatbot, doc_filter, auto_route, num_chunks],
        outputs=[chatbot]
    ).then(
        fn=update_status_bar,
        outputs=[status_bar]
    )

    pdf_input.change(
        fn=process_pdf_with_status,
        inputs=[pdf_input],
        outputs=[status_output, structure_output, doc_filter, status_bar]
    )

demo.launch(debug=True)
