// For Showing Login / Sign up modal
const openLoginBtn = document.getElementById("openLoginBtn");
const closeLoginBtn = document.getElementById("closeLoginBtn");
const modalLogin = document.getElementById("modalLogin");

openLoginBtn.addEventListener("click", function() {
    modalLogin.style.display = "block";
});

closeLoginBtn.addEventListener("click", function() {
    console.log("close btn clik")
    modalLogin.style.display = "none";
});

const openSignupBtn = document.getElementById("openSignupBtn");
const closeSignupBtn = document.getElementById("closeSignupBtn");
const modalSignup = document.getElementById("modalSignup");

openSignupBtn.addEventListener("click", function() {
    modalSignup.style.display = "block";
});

closeSignupBtn.addEventListener("click", function() {
    console.log("close btn clik")
    modalSignup.style.display = "none";
});
