import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import asyncio
import threading
from email_system import send_b2b_sales_emails

class RecipientRow:
    def __init__(self, parent, remove_callback):
        self.frame = ttk.Frame(parent)
        self.name_var = tk.StringVar()
        self.email_var = tk.StringVar()
        self.website_var = tk.StringVar()
        self.name_entry = ttk.Entry(self.frame, textvariable=self.name_var, width=18)
        self.email_entry = ttk.Entry(self.frame, textvariable=self.email_var, width=22)
        self.website_entry = ttk.Entry(self.frame, textvariable=self.website_var, width=28)
        self.remove_btn = ttk.Button(self.frame, text="Remove", command=self.remove, style="Danger.TButton")
        self.remove_callback = remove_callback
        self.name_entry.grid(row=0, column=0, padx=2, pady=2)
        self.email_entry.grid(row=0, column=1, padx=2, pady=2)
        self.website_entry.grid(row=0, column=2, padx=2, pady=2)
        self.remove_btn.grid(row=0, column=3, padx=2, pady=2)
    def grid(self, **kwargs):
        self.frame.grid(**kwargs)
    def remove(self):
        self.frame.destroy()
        self.remove_callback(self)
    def get(self):
        return {
            "name": self.name_var.get().strip(),
            "email": self.email_var.get().strip(),
            "website": self.website_var.get().strip(),
        }

class B2BEmailGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ComplAI B2B Sales Email Sender")
        self.root.geometry("750x600")
        self.root.configure(bg="#f7f7fa")
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Danger.TButton", foreground="#fff", background="#e74c3c")
        style.map("Danger.TButton", background=[('active', '#c0392b')])
        style.configure("TLabel", background="#f7f7fa", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), background="#f7f7fa", foreground="#2d3436")
        # Header
        ttk.Label(root, text="ComplAI B2B Sales Email Sender", style="Header.TLabel").pack(pady=(18, 2))
        ttk.Label(root, text="Send personalized cold emails to companies in seconds!", style="TLabel").pack(pady=(0, 12))
        # Sender name
        sender_frame = ttk.Frame(root)
        sender_frame.pack(pady=4)
        ttk.Label(sender_frame, text="Sender Name:").grid(row=0, column=0, sticky="e", padx=2)
        self.sender_var = tk.StringVar()
        self.sender_var.set("Itamar Zam")
        ttk.Entry(sender_frame, textvariable=self.sender_var, width=30).grid(row=0, column=1, padx=4)
        # Message
        msg_frame = ttk.Frame(root)
        msg_frame.pack(pady=4)
        ttk.Label(msg_frame, text="Message:").grid(row=0, column=0, sticky="ne", padx=2)
        self.msg_text = tk.Text(msg_frame, width=60, height=3, font=("Segoe UI", 10))
        self.msg_text.insert("1.0", "Send a cold sales email to the following recipients.")
        self.msg_text.grid(row=0, column=1, padx=4)
        # Recipients
        rec_frame = ttk.LabelFrame(root, text="Recipients")
        rec_frame.pack(pady=10, fill="x", padx=16)
        header = ttk.Frame(rec_frame)
        header.grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Company Name", width=18).grid(row=0, column=0)
        ttk.Label(header, text="Email", width=22).grid(row=0, column=1)
        ttk.Label(header, text="Website", width=28).grid(row=0, column=2)
        self.recipient_rows = []
        self.recipient_container = ttk.Frame(rec_frame)
        self.recipient_container.grid(row=1, column=0, sticky="w")
        self.add_recipient_row()
        add_btn = ttk.Button(rec_frame, text="Add Recipient", command=self.add_recipient_row)
        add_btn.grid(row=2, column=0, pady=6, sticky="w")
        # Send button
        self.send_btn = ttk.Button(root, text="Send Emails", command=self.on_send, style="TButton")
        self.send_btn.pack(pady=16)
        # Output
        out_frame = ttk.LabelFrame(root, text="Output / Status")
        out_frame.pack(fill="both", expand=True, padx=16, pady=8)
        self.output_box = scrolledtext.ScrolledText(out_frame, height=10, font=("Consolas", 10), wrap="word", state="disabled", bg="#f7f7fa")
        self.output_box.pack(fill="both", expand=True, padx=4, pady=4)
    def add_recipient_row(self):
        row = RecipientRow(self.recipient_container, self.remove_recipient_row)
        row.grid(row=len(self.recipient_rows), column=0, pady=2, sticky="w")
        self.recipient_rows.append(row)
    def remove_recipient_row(self, row):
        self.recipient_rows.remove(row)
    def on_send(self):
        message = self.msg_text.get("1.0", "end").strip()
        sender_name = self.sender_var.get().strip()
        recipients = [r.get() for r in self.recipient_rows if r.get()["email"] and r.get()["name"]]
        if not message or not sender_name or not recipients:
            messagebox.showerror("Missing Data", "Please fill in all fields and add at least one recipient.")
            return
        self.send_btn.config(state="disabled")
        self.output_box.config(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.insert("end", "Sending emails... Please wait.\n")
        self.output_box.config(state="disabled")
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(send_b2b_sales_emails(message, recipients, sender_name))
                self.show_result(result)
            except Exception as e:
                self.show_result(f"ERROR: {e}")
            finally:
                self.send_btn.config(state="normal")
        threading.Thread(target=run_async, daemon=True).start()
    def show_result(self, result):
        self.output_box.config(state="normal")
        self.output_box.insert("end", f"\n{result}\n")
        self.output_box.see("end")
        self.output_box.config(state="disabled")

def main():
    root = tk.Tk()
    app = B2BEmailGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 