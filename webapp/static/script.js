// ══════════════════════════════════════════════════════════════════════════════
// EduPredict.AI — Frontend Logic
// Handles tab switching, form submission, API calls, animations, and particles
// ══════════════════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    initTabs();
    initForms();
    animateCounters();
});

// ──────────────────────────────────────────────────────────────────────────────
// Floating Particles
// ──────────────────────────────────────────────────────────────────────────────

function initParticles() {
    const container = document.getElementById("particles");
    const count = 30;

    for (let i = 0; i < count; i++) {
        const particle = document.createElement("div");
        particle.classList.add("particle");
        particle.style.left = Math.random() * 100 + "%";
        particle.style.animationDuration = 8 + Math.random() * 14 + "s";
        particle.style.animationDelay = Math.random() * 10 + "s";
        particle.style.width = particle.style.height = 2 + Math.random() * 4 + "px";
        particle.style.background = `hsl(${250 + Math.random() * 40}, 80%, ${60 + Math.random() * 20}%)`;
        container.appendChild(particle);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tab Navigation
// ──────────────────────────────────────────────────────────────────────────────

function initTabs() {
    const buttons = document.querySelectorAll(".tab-btn");
    const panels = document.querySelectorAll(".tab-panel");

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;

            // Update button states
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Update panel visibility
            panels.forEach(p => p.classList.remove("active"));
            document.getElementById(`panel-${target}`).classList.add("active");
        });
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Animated Counters (Hero Stats)
// ──────────────────────────────────────────────────────────────────────────────

function animateCounters() {
    const counters = document.querySelectorAll(".stat-number");

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.dataset.count, 10);
                animateNumber(el, 0, target, 1200);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(c => observer.observe(c));
}

function animateNumber(el, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = Math.round(start + (end - start) * eased);
        el.textContent = value.toLocaleString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ──────────────────────────────────────────────────────────────────────────────
// Form Handling
// ──────────────────────────────────────────────────────────────────────────────

function initForms() {
    // Persistence form
    const persistenceForm = document.getElementById("form-persistence");
    persistenceForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await handlePersistence(persistenceForm);
    });

    // GPA form
    const gpaForm = document.getElementById("form-gpa");
    gpaForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await handleGPA(gpaForm);
    });

    // Improvement form
    const improvementForm = document.getElementById("form-improvement");
    improvementForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await handleImprovement(improvementForm);
    });
}

// ── Helper: collect form data as object ──
function getFormData(form) {
    const fd = new FormData(form);
    const data = {};
    for (const [key, value] of fd.entries()) {
        data[key] = value;
    }
    return data;
}

// ── Helper: toggle loading state ──
function setLoading(btn, loading) {
    if (loading) {
        btn.classList.add("loading");
        btn.disabled = true;
    } else {
        btn.classList.remove("loading");
        btn.disabled = false;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Persistence Prediction
// ──────────────────────────────────────────────────────────────────────────────

async function handlePersistence(form) {
    const btn = document.getElementById("submit-persistence");
    const resultsEl = document.getElementById("results-persistence");

    setLoading(btn, true);

    try {
        const data = getFormData(form);
        const response = await fetch("/predict/persistence", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || "Prediction failed");
        }

        // Show results
        resultsEl.classList.remove("hidden");

        // NN gauge
        const nnPct = Math.round(result.nn.probability * 100);
        animateGauge("gauge-nn", nnPct);
        document.getElementById("gauge-nn-value").textContent = nnPct + "%";

        const nnLabel = document.getElementById("label-nn");
        nnLabel.textContent = result.nn.label;
        nnLabel.className = "result-label " + (result.nn.prediction === 1 ? "persist" : "at-risk");

        // RF gauge
        const rfPct = Math.round(result.rf.probability * 100);
        animateGauge("gauge-rf", rfPct);
        document.getElementById("gauge-rf-value").textContent = rfPct + "%";

        const rfLabel = document.getElementById("label-rf");
        rfLabel.textContent = result.rf.label;
        rfLabel.className = "result-label " + (result.rf.prediction === 1 ? "persist" : "at-risk");

        // Scroll into view
        resultsEl.scrollIntoView({ behavior: "smooth", block: "center" });

    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        setLoading(btn, false);
    }
}

function animateGauge(id, pct) {
    const el = document.getElementById(id);
    const circumference = 2 * Math.PI * 50; // r=50
    const offset = circumference - (pct / 100) * circumference;

    // Reset first for re-animation
    el.style.transition = "none";
    el.style.strokeDashoffset = circumference;

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            el.style.transition = "stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)";
            el.style.strokeDashoffset = offset;
        });
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GPA Prediction
// ──────────────────────────────────────────────────────────────────────────────

async function handleGPA(form) {
    const btn = document.getElementById("submit-gpa");
    const resultsEl = document.getElementById("results-gpa");

    setLoading(btn, true);

    try {
        const data = getFormData(form);
        const response = await fetch("/predict/gpa", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || "Prediction failed");
        }

        resultsEl.classList.remove("hidden");

        // NN GPA
        const nnGpa = result.nn.predicted_gpa;
        document.getElementById("gpa-nn-value").textContent = nnGpa.toFixed(2);
        animateBar("gpa-nn-bar", nnGpa, 4.5);

        // RF GPA
        const rfGpa = result.rf.predicted_gpa;
        document.getElementById("gpa-rf-value").textContent = rfGpa.toFixed(2);
        animateBar("gpa-rf-bar", rfGpa, 4.5);

        resultsEl.scrollIntoView({ behavior: "smooth", block: "center" });

    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        setLoading(btn, false);
    }
}

function animateBar(id, value, max) {
    const bar = document.getElementById(id);
    const pct = Math.min((value / max) * 100, 100);

    bar.style.transition = "none";
    bar.style.width = "0%";

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            bar.style.transition = "width 1s cubic-bezier(0.4, 0, 0.2, 1)";
            bar.style.width = pct + "%";
        });
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GPA Improvement Prediction
// ──────────────────────────────────────────────────────────────────────────────

async function handleImprovement(form) {
    const btn = document.getElementById("submit-improvement");
    const resultsEl = document.getElementById("results-improvement");

    setLoading(btn, true);

    try {
        const data = getFormData(form);
        const response = await fetch("/predict/improvement", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || "Prediction failed");
        }

        resultsEl.classList.remove("hidden");

        const val = result.predicted_improvement;
        const display = document.getElementById("improvement-value");

        // Format with + or - sign
        const sign = val >= 0 ? "+" : "";
        display.textContent = sign + val.toFixed(2);

        // Color coding
        display.className = "improvement-display";
        if (val > 0.05) {
            display.classList.add("positive");
        } else if (val < -0.05) {
            display.classList.add("negative");
        } else {
            display.classList.add("neutral");
        }

        // Update caption
        const caption = document.getElementById("improvement-caption");
        if (val > 0.05) {
            caption.textContent = "📈 GPA is expected to improve";
        } else if (val < -0.05) {
            caption.textContent = "📉 GPA is expected to decline";
        } else {
            caption.textContent = "➡️ GPA is expected to remain stable";
        }

        resultsEl.scrollIntoView({ behavior: "smooth", block: "center" });

    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        setLoading(btn, false);
    }
}
