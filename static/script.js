/* ===============================
   ACCURACY BAR CHART (PERFORMANCE ANALYSIS)
================================ */

const acc = document.getElementById("accuracyChart");

if (acc) {
  new Chart(acc, {
    type: "bar",
    data: {
      labels: ["Logistic Regression", "SVM", "Random Forest", "XGBoost (Best)"],
      datasets: [
        {
          label: "Accuracy (%)",
          data: [65, 68, 71, 74],
          backgroundColor: [
            "var(--primary-color)", // Logistic
            "var(--warning-color)", // SVM
            "var(--accent-color)", // Random Forest
            "var(--success-color)", // XGBoost (BEST)
          ],
          borderRadius: 8,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 2000,
        easing: "easeOutBounce",
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: "var(--card-bg)",
          titleColor: "var(--text-primary)",
          bodyColor: "var(--text-secondary)",
          borderColor: "var(--border-color)",
          borderWidth: 1,
          callbacks: {
            label: function (context) {
              if (context.dataIndex === 3) {
                return "XGBoost (Best Model): " + context.raw + "%";
              }
              return context.raw + "%";
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: {
            color: "var(--border-color)",
          },
          ticks: {
            color: "var(--text-secondary)",
            callback: function(value) {
              return value + "%";
            }
          },
          title: {
            display: true,
            text: "Accuracy Score",
            color: "var(--text-primary)",
          },
        },
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: "var(--text-secondary)",
          },
          title: {
            display: true,
            text: "Machine Learning Algorithms",
            color: "var(--text-primary)",
          },
        },
      },
    },
  });
}

/* ===============================
   FEATURE IMPORTANCE RADAR CHART
================================ */

const heat = document.getElementById("heatmapChart");

if (heat) {
  new Chart(heat, {
    type: "radar",
    data: {
      labels: [
        "Age",
        "Systolic BP",
        "Diastolic BP",
        "BMI",
        "Cholesterol",
        "Glucose",
        "Smoking",
        "Alcohol",
        "Activity",
      ],
      datasets: [
        {
          label: "Feature Importance",
          data: [0.85, 0.82, 0.78, 0.65, 0.58, 0.52, 0.45, 0.38, 0.35],
          backgroundColor: "rgba(59, 130, 246, 0.2)",
          borderColor: "var(--primary-color)",
          borderWidth: 3,
          pointBackgroundColor: "var(--primary-color)",
          pointBorderColor: "var(--card-bg)",
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 2000,
        easing: "easeOutQuart",
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 1,
          ticks: {
            stepSize: 0.2,
            color: "var(--text-secondary)",
            callback: function(value) {
              return (value * 100) + "%";
            }
          },
          grid: {
            color: "var(--border-color)",
          },
          angleLines: {
            color: "var(--border-color)",
          },
          pointLabels: {
            color: "var(--text-primary)",
            font: {
              size: 12,
              weight: "500",
            }
          }
        },
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: "var(--card-bg)",
          titleColor: "var(--text-primary)",
          bodyColor: "var(--text-secondary)",
          borderColor: "var(--border-color)",
          borderWidth: 1,
          callbacks: {
            label: function (context) {
              return context.label + ": " + (context.raw * 100).toFixed(1) + "%";
            },
          },
        },
      },
    },
  });
}

/* ===============================
   FAQ ACCORDION FUNCTIONALITY
================================ */

function initFAQAccordion() {
  const faqItems = document.querySelectorAll('.faq-item');

  faqItems.forEach(item => {
    const question = item.querySelector('.faq-question');
    const answer = item.querySelector('.faq-answer');
    const toggle = item.querySelector('.faq-toggle');

    question.addEventListener('click', () => {
      const isOpen = item.classList.contains('open');

      // Close all FAQ items
      faqItems.forEach(otherItem => {
        otherItem.classList.remove('open');
        otherItem.querySelector('.faq-answer').style.maxHeight = null;
      });

      // Open clicked item if it wasn't already open
      if (!isOpen) {
        item.classList.add('open');
        answer.style.maxHeight = answer.scrollHeight + 'px';
      }
    });
  });
}

/* ===============================
   FAQ SEARCH FUNCTIONALITY
================================ */

function initFAQSearch() {
  const searchInput = document.getElementById('faqSearch');
  const faqItems = document.querySelectorAll('.faq-item');

  if (!searchInput) return;

  searchInput.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase().trim();

    faqItems.forEach(item => {
      const question = item.querySelector('h4').textContent.toLowerCase();
      const answer = item.querySelector('.faq-answer').textContent.toLowerCase();

      if (question.includes(searchTerm) || answer.includes(searchTerm) || searchTerm === '') {
        item.style.display = 'block';
      } else {
        item.style.display = 'none';
      }
    });
  });
}

/* ===============================
   FAQ CATEGORY FILTERING
================================ */

function initFAQCategories() {
  const categoryBtns = document.querySelectorAll('.category-btn');
  const faqItems = document.querySelectorAll('.faq-item');

  categoryBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active class from all buttons
      categoryBtns.forEach(b => b.classList.remove('active'));
      // Add active class to clicked button
      btn.classList.add('active');

      const category = btn.dataset.category;

      faqItems.forEach(item => {
        if (category === 'all' || item.dataset.category === category) {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      });
    });
  });
}

/* ===============================
   DOM CONTENT LOADED
================================ */

document.addEventListener("DOMContentLoaded", () => {
  // Theme toggle functionality
  const toggleBtn = document.getElementById("themeToggle");

  // Apply saved theme on every page load
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "light") {
    document.body.classList.add("light-mode");
  }

  if (toggleBtn) {
    // Set icon based on current theme
    toggleBtn.textContent = document.body.classList.contains("light-mode")
      ? "☀️"
      : "🌙";

    toggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("light-mode");

      const isLight = document.body.classList.contains("light-mode");
      toggleBtn.textContent = isLight ? "☀️" : "🌙";

      // Save the theme
      localStorage.setItem("theme", isLight ? "light" : "dark");
    });
  }

  // Initialize all interactive features
  createParticles();
  initScrollAnimations();
  initFormLoading();
  initMicroInteractions();
  initFAQAccordion();
  initFAQSearch();
  initFAQCategories();
});

// CREATE FLOATING PARTICLES
function createParticles() {
  const particlesContainer = document.getElementById('particles');
  if (!particlesContainer) return;

  for (let i = 0; i < 50; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.animationDelay = Math.random() * 20 + 's';
    particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
    particle.style.width = particle.style.height = Math.random() * 6 + 2 + 'px';
    particlesContainer.appendChild(particle);
  }
}

// SCROLL ANIMATIONS
function initScrollAnimations() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);

  document.querySelectorAll('.scroll-fade-in').forEach(el => {
    observer.observe(el);
  });
}

// FORM LOADING ANIMATION
function initFormLoading() {
  const form = document.getElementById('predictionForm');
  const submitBtn = document.getElementById('submitBtn');

  if (!form || !submitBtn) return;

  form.addEventListener('submit', (e) => {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>🔄</span> Analyzing...';

    // Add loading class for additional styling
    submitBtn.classList.add('loading');
  });
}

// MICRO INTERACTIONS
function initMicroInteractions() {
  // Enhanced card hover effects
  document.querySelectorAll('.info-card, .chart-card, .result-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
      card.style.transform = 'translateY(-8px) scale(1.02)';
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'translateY(0) scale(1)';
    });
  });

  // Button ripple effect
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      ripple.className = 'ripple-effect';
      ripple.style.left = (e.offsetX - 10) + 'px';
      ripple.style.top = (e.offsetY - 10) + 'px';
      this.appendChild(ripple);
      setTimeout(() => ripple.remove(), 600);
    });
  });

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
}

// RIPPLE ANIMATION STYLES
const style = document.createElement('style');
style.textContent = `
@keyframes ripple {
  to {
    transform: scale(4);
    opacity: 0;
  }
}

.ripple-effect {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.6);
  transform: scale(0);
  animation: ripple 0.6s linear;
  width: 20px;
  height: 20px;
  pointer-events: none;
}

.btn.loading {
  pointer-events: none;
  position: relative;
}

.btn.loading::after {
  content: '';
  position: absolute;
  width: 16px;
  height: 16px;
  margin: auto;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
`;
document.head.appendChild(style);


