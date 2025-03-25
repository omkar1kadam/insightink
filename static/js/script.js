document.addEventListener("DOMContentLoaded", function () {
    const chatInput = document.querySelector(".chat-input");
    const chatOutput = document.querySelector(".chatbot-output");
    const chatSend = document.querySelector(".chat-send");

    chatSend.addEventListener("click", () => {
        if (chatInput.value.trim() !== "") {
            chatOutput.innerText = "You said: " + chatInput.value;
            chatInput.value = ""; // Clear input after sending
        }
    });
});
