window.addEventListener("load", () => {
  var SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();

  var agendaNumber = 0;
  var noteContent = ["", "", ""];

  recognition.continuous = true;

  recognition.onresult = function (event) {
    var current = event.resultIndex;

    // Get a transcript of what was said.
    var transcript = event.results[current][0].transcript;
    if (agendaNumber == 1) {
    }
    noteContent[agendaNumber] += transcript;
    console.log(noteContent);
  };

  recognition.onstart = function () {
    console.log(
      "Voice recognition activated. Try speaking into the microphone."
    );
  };

  recognition.onspeechend = function () {
    console.log(
      "You were quiet for a while so voice recognition turned itself off."
    );
  };

  recognition.onerror = function (event) {
    if (event.error == "no-speech") {
      console.log("No speech was detected. Try again.");
    }
  };

  document.getElementById("stop-recording").addEventListener("click", (e) => {
    recognition.stop();
    console.log("Recording stopped");
  });

  document.getElementById("agenda1").addEventListener("click", (e) => {
    e.preventDefault();
    agendaNumber = 0;
    if (noteContent[0].length) {
      noteContent[0] += " ";
    }
    recognition.start();
  });
  document.getElementById("agenda2").addEventListener("click", (e) => {
    e.preventDefault();
    agendaNumber = 1;
    if (noteContent[0].length) {
      noteContent[0] += " ";
    }
    recognition.start();
  });
  document.getElementById("agenda3").addEventListener("click", (e) => {
    e.preventDefault();
    agendaNumber = 3;
    if (noteContent[0].length) {
      noteContent[0] += " ";
    }
    recognition.start();
  });
});
