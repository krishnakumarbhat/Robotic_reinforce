/**
 * LeevAI Website - Main JavaScript
 * Handles loading animation, navigation, and interactions
 */

class LeevAIWebsite {
    constructor() {
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        this.loadingAnimation();
        this.setupNavigation();
        this.setupScrollEffects();
    }

    /**
     * Loading Animation - Shows logo animation then hides loading screen
     */
    loadingAnimation() {
        const intro = document.querySelector('.intro');
        const logoSpans = document.querySelectorAll('.logo-header .logo');

        if (!intro || !logoSpans.length) return;

        // Animate logo letters
        logoSpans.forEach((span, index) => {
            setTimeout(() => {
                span.classList.add('active');
            }, (index + 1) * 400);
        });

        // Fade out logo letters
        setTimeout(() => {
            logoSpans.forEach((span, index) => {
                setTimeout(() => {
                    span.classList.remove('active');
                    span.classList.add('fade');
                }, (index + 1) * 50);
            });
        }, 2000);

        // Hide loading screen
        setTimeout(() => {
            intro.style.top = '-100vh';
        }, 2300);
    }

    /**
     * Navigation Setup - Mobile menu and scroll effects
     */
    setupNavigation() {
        const navbar = document.querySelector('.navbar');
        const menuToggle = document.querySelector('.menu-toggle');
        const mobileOverlay = document.querySelector('.mobile-menu-overlay');
        const closeBtn = document.querySelector('.close-btn');

        // Mobile menu toggle
        if (menuToggle && mobileOverlay) {
            menuToggle.addEventListener('click', () => {
                mobileOverlay.classList.add('active');
                document.body.style.overflow = 'hidden';
            });

            const closeMobileMenu = () => {
                mobileOverlay.classList.remove('active');
                document.body.style.overflow = '';
            };

            closeBtn?.addEventListener('click', closeMobileMenu);
            
            // Close on overlay click
            mobileOverlay.addEventListener('click', (e) => {
                if (e.target === mobileOverlay) {
                    closeMobileMenu();
                }
            });

            // Close on escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && mobileOverlay.classList.contains('active')) {
                    closeMobileMenu();
                }
            });
        }

        // Navbar scroll effect
        if (navbar) {
            let lastScrollY = window.scrollY;
            
            window.addEventListener('scroll', () => {
                const scrollY = window.scrollY;
                
                if (scrollY > 50) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
                
                lastScrollY = scrollY;
            });
        }
    }

    /**
     * Scroll Effects - Smooth scrolling and animations
     */
    setupScrollEffects() {
        // Smooth scroll for anchor links
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

        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    }

    /**
     * Utility function to navigate to different pages
     */
    static navigateTo(page) {
        const routes = {
            'home': 'index.html',
            'about': 'pages/about/index.html',
            'blog': 'pages/blog/index.html',
            'careers': 'pages/careers/index.html'
        };

        if (routes[page]) {
            window.location.href = routes[page];
        }
    }
}

// Initialize the website
new LeevAIWebsite();

// Export for use in other scripts
window.LeevAI = LeevAIWebsite;
