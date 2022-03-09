function count_word(val) {
  var wom = val.match(/\S+/g);
  return {
    words: wom ? wom.length : 0,
  };
}

var textContent = document.getElementById("text");
var showWordCount = document.getElementById("countWord");

textContent.addEventListener(
  "input",
  function () {
    var v = count_word(this.value);
    showWordCount.innerHTML = "Words: " + v.words;
  },
  false
);

var textContent2 = document.getElementById("text2");
var showWordCount2 = document.getElementById("countWord2");

var buttonClear = document.querySelector("#clear");
var clearNotif = document.querySelector("#clearNoti");

buttonClear.addEventListener(
  "click",
  function () {
    textContent.value = "";
    textContent2.value = "";
    showWordCount.innerHTML = "Words: " + 0;
    showWordCount2.innerHTML = "Words: " + 0;
    clearNotif.classList.add("show");
    setTimeout(() => {
      clearNotif.classList.remove("show");
    }, 1000);
  },
  false
);

var buttonCopy = document.getElementById("copy");
var copyNotif = document.querySelector("#copyNoti");

buttonCopy.addEventListener("click", function (event) {
  event.preventDefault();
  textContent2.select();
  document.execCommand("copy");
  textContent2.setSelectionRange(0, 0);
  textContent2.blur();
  copyNotif.classList.add("show");
  setTimeout(() => {
    copyNotif.classList.remove("show");
  }, 1000);
});

var modal = document.getElementById("modal");
var buttonModal = document.getElementById("manual");
// var span = document.getElementsByClassName("close")[0];
buttonModal.addEventListener("click", function (event) {
  modal.classList.add("show");
});

window.addEventListener(
  "click",
  function (e) {
    modal.classList.remove("show");
  },
  true
);

var buttonTranslate = document.getElementById("translateButton");
var translating = document.getElementById("translating");
buttonTranslate.addEventListener("click", function (e) {
  textContent2.value = textContent.value;
  var temp = textContent2.value;
  showWordCount2.innerHTML = "Words: " + temp.split(" ").length;
  buttonTranslate.style.display = "none";
  translating.style.display = "flex";
  textContent2.classList.add("show");
});

var domain = document.getElementById("domainChoice").value;

$(document).on("submit", "#translatorForm", function (e) {
  console.log("hello");
  e.preventDefault();
  $.ajax({
    type: "POST",
    url: "/",
    data: {
      textVal: $("#text").val(),
      userDomain: domain,
    },
    success: function (response) {
        textContent2.value = JSON.stringify(response).replace(/\"/g, "");
        textContent2.classList.remove("show");
        buttonTranslate.innerHTML = "Translate";
        buttonTranslate.style.display = "flex";
        translating.style.display = "none";
    },
    failure: function (response) {
      alert("failure");
    },
  });
});

tailwind.config = {
  theme: {
    extend: {
      screens: {
        sm: { min: "0px", max: "767px" },
        md: { min: "768px", max: "1023px" },
        lg: "1024px",
        xl: "1280px",
        "2xl": "1536px",
      },
    },
  },
};
