(function() {
    'use strict';

    document.addEventListener('DOMContentLoaded', function() {
        const links = document.querySelectorAll('.nav-link');
        const path = window.location.pathname;
        links.forEach(l => {
            if (l.getAttribute('href') === path) l.classList.add('active');
        });

        const predictsBtn = document.getElementById('predictBtn');
        if (predictsBtn) {
            predictsBtn.addEventListener('click', function() {
                this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                this.disabled = true;
            });
        }

        const meterFills = document.querySelectorAll('.meter-fill');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(e => {
                if (e.isIntersecting) {
                    e.target.style.width = e.target.style.width;
                }
            });
        });
        meterFills.forEach(el => observer.observe(el));
    });

})();
