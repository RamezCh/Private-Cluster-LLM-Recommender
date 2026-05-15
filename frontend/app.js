const API_BASE = '';

class Carousel {
    constructor(containerId, counterId) {
        this.container = document.getElementById(containerId);
        this.wrapper = this.container.querySelector('.carousel-wrapper');
        this.track = this.container.querySelector('.carousel-track');
        this.dotsContainer = this.container.parentElement.querySelector('.carousel-dots');
        this.counterElement = document.getElementById(counterId);
        this.currentIndex = 0;
        this.cards = [];
        this.touchStartX = 0;
        this.touchEndX = 0;
    }

    init() {
        this.updateCards();
        this.createDots();
        this.bindEvents();
        this.updateCarousel();
    }

    updateCards() {
        this.cards = Array.from(this.track.querySelectorAll('.card'));
    }

    createDots() {
        if (!this.dotsContainer) return;
        this.dotsContainer.innerHTML = '';
        const numDots = Math.max(1, this.cards.length);
        for (let i = 0; i < numDots; i++) {
            const dot = document.createElement('button');
            dot.className = 'dot';
            dot.setAttribute('aria-label', `Go to slide ${i + 1}`);
            dot.addEventListener('click', () => this.goTo(i));
            this.dotsContainer.appendChild(dot);
        }
    }

    bindEvents() {
        const prevBtn = this.container.querySelector('.carousel-prev');
        const nextBtn = this.container.querySelector('.carousel-next');
        if (prevBtn) prevBtn.addEventListener('click', () => this.prev());
        if (nextBtn) nextBtn.addEventListener('click', () => this.next());

        this.track.addEventListener('touchstart', e => {
            this.touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });
        this.track.addEventListener('touchend', e => {
            this.touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe();
        }, { passive: true });

        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'ArrowLeft') this.prev();
            if (e.key === 'ArrowRight') this.next();
        });
    }

    handleSwipe() {
        const diff = this.touchStartX - this.touchEndX;
        if (Math.abs(diff) > 50) {
            if (diff > 0) this.next();
            else this.prev();
        }
    }

    goTo(index) {
        const numCards = this.cards.length;
        if (numCards === 0) return;
        this.currentIndex = Math.max(0, Math.min(index, numCards - 1));
        this.updateCarousel();
    }

    prev() {
        const numCards = this.cards.length;
        if (numCards === 0) return;
        this.currentIndex = (this.currentIndex - 1 + numCards) % numCards;
        this.updateCarousel();
    }

    next() {
        const numCards = this.cards.length;
        if (numCards === 0) return;
        this.currentIndex = (this.currentIndex + 1) % numCards;
        this.updateCarousel();
    }

    updateCarousel() {
        if (this.cards.length === 0) return;
        
        const containerWidth = this.wrapper.offsetWidth || 800;
        const cardWidth = 260;
        const activeWidth = 320;
        const gap = 16;
        
        const centerOffset = (containerWidth - activeWidth) / 2;
        const scrollToIndex = this.currentIndex * (cardWidth + gap);
        const offset = centerOffset - scrollToIndex;
        
        this.track.style.transform = `translateX(${offset}px)`;

        this.cards.forEach((card, i) => {
            card.classList.toggle('active', i === this.currentIndex);
        });

        const dots = this.dotsContainer?.querySelectorAll('.dot');
        dots?.forEach((dot, i) => {
            dot.classList.toggle('active', i === this.currentIndex);
        });

        this.updateCounter();
    }

    updateCounter() {
        if (this.counterElement && this.cards.length > 0) {
            this.counterElement.textContent = `${this.currentIndex + 1} / ${this.cards.length}`;
        } else if (this.counterElement) {
            this.counterElement.textContent = '0 / 0';
        }
    }

    setCards(html) {
        this.track.innerHTML = html;
        this.updateCards();
        this.createDots();
        this.currentIndex = 0;
        setTimeout(() => this.updateCarousel(), 50);
    }
}

let showcaseCarousel = null;
let resultsCarousel = null;

function showToast(message, type = 'error') {
    const toast = document.getElementById('toast') || createToast();
    toast.textContent = message;
    toast.className = `toast visible ${type}`;
    setTimeout(() => toast.classList.remove('visible'), 4000);
}

function createToast() {
    const toast = document.createElement('div');
    toast.id = 'toast';
    toast.className = 'toast';
    document.body.appendChild(toast);
    return toast;
}

function formatNumber(num) {
    if (num == null) return 'N/A';
    return Number(num).toLocaleString();
}

function createCardHTML(item) {
    const model = item.model || item || {};
    const hardware = item.hardware || {};
    const modelName = model.base_model || model.model_id || 'Unknown Model';
    const provider = model.model_type || 'Open-weight';
    
    return `
        <div class="card">
            <span class="card-category">${item.category || item.label || 'Model'}</span>
            <h3 class="card-title">${modelName}</h3>
            <p class="card-provider">${provider}</p>
            <div class="card-stats">
                <div class="stat-item">
                    <div class="stat-label">Parameters</div>
                    <div class="stat-value">${formatNumber(model.params_billions)}B</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Quantization</div>
                    <div class="stat-value">${model.hosting_strategy || 'FP16'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">VRAM (FP16)</div>
                    <div class="stat-value">${model.vram_fp16_gb || '?'}GB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Score</div>
                    <div class="stat-value">${model.scores?.final ? (model.scores.final * 100).toFixed(0) + '%' : 'N/A'}</div>
                </div>
            </div>
            ${model.hf_repo_id ? `<a href="https://huggingface.co/${model.hf_repo_id}" target="_blank" class="card-link">View on HuggingFace</a>` : ''}
            <div class="card-details">
                <h4>Benchmarks</h4>
                <div class="benchmark-grid">
                    ${createBenchmarkHTML(model.benchmarks)}
                </div>
            </div>
        </div>
    `;
}

function createBenchmarkHTML(benchmarks) {
    if (!benchmarks) return '<p>No benchmark data</p>';
    return Object.entries(benchmarks).map(([name, score]) => `
        <div class="benchmark-item">
            <div class="benchmark-name">${name}</div>
            <div class="benchmark-score">${typeof score === 'number' ? score.toFixed(1) : 'N/A'}</div>
        </div>
    `).join('');
}

function attachCardListeners() {
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.closest('.card-link')) return;
            const details = card.querySelector('.card-details');
            if (details) {
                if (details.classList.contains('visible')) {
                    details.classList.remove('visible');
                } else {
                    document.querySelectorAll('.card-details.visible').forEach(d => d.classList.remove('visible'));
                    details.classList.add('visible');
                }
            }
        });
    });
}

function showLoadingSkeletons(track) {
    track.innerHTML = Array(3).fill(`
        <div class="card">
            <div style="background: var(--bg-secondary); border-radius: 6px; height: 1.2rem; width: 60%; margin-bottom: 0.75rem;"></div>
            <div style="background: var(--bg-secondary); border-radius: 6px; height: 1.4rem; width: 85%; margin-bottom: 0.5rem;"></div>
            <div style="background: var(--bg-secondary); border-radius: 6px; height: 0.9rem; width: 50%; margin-bottom: 1rem;"></div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;">
                ${Array(4).fill('<div style="background: var(--bg-secondary); padding: 0.7rem; border-radius: 8px;"></div>').join('')}
            </div>
        </div>
    `).join('');
}

async function fetchShowcase() {
    const track = document.querySelector('#showcase-carousel .carousel-track');
    showLoadingSkeletons(track);
    showcaseCarousel?.init();

    try {
        const response = await fetch(`${API_BASE}/api/showcase`);
        const data = await response.json();
        
        if (data.success && data.showcase?.length > 0) {
            track.innerHTML = data.showcase.map(item => createCardHTML(item)).join('');
            showcaseCarousel = new Carousel('showcase-carousel', 'showcase-counter');
            showcaseCarousel.init();
            attachCardListeners();
        } else {
            track.innerHTML = `
                <div class="empty-state">
                    <p>Select your hardware and get personalized recommendations</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Showcase fetch error:', error);
        track.innerHTML = `
            <div class="empty-state">
                <p>Failed to load showcase</p>
            </div>
        `;
    }
}

async function fetchRecommendations() {
    const gpuSelect = document.getElementById('gpu-select');
    const gpuCount = document.getElementById('gpu-count');
    const topKSelect = document.getElementById('top-k-select');
    const useCase = getSelectedUseCases();
    
    const selectedOption = gpuSelect.options[gpuSelect.selectedIndex];
    const gpuName = selectedOption?.text || '';
    const count = parseInt(gpuCount?.value) || 1;
    const topK = parseInt(topKSelect?.value) || 5;
    
    if (!gpuName || gpuName === '-- SELECT GPU --') {
        showToast('Please select a GPU type');
        return;
    }
    
    const hardwareText = count > 1 ? `${count}x ${gpuName}` : gpuName;

    const btn = document.getElementById('recommend-btn');
    btn.disabled = true;
    btn.textContent = 'Getting recommendations...';

    const track = document.querySelector('#results-carousel .carousel-track');
    showLoadingSkeletons(track);

    const showcaseSection = document.getElementById('showcase-section');
    const resultsSection = document.getElementById('results-section');
    showcaseSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    resultsCarousel = new Carousel('results-carousel', 'results-counter');
    resultsCarousel.init();

    try {
        const response = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hardware_text: hardwareText, use_case: useCase, top_k: topK })
        });
        
        const data = await response.json();
        
        if (data.success && data.recommendations?.length > 0) {
            track.innerHTML = data.recommendations.map(item => createCardHTML(item)).join('');
            resultsCarousel = new Carousel('results-carousel', 'results-counter');
            resultsCarousel.init();
            attachCardListeners();
            
            document.getElementById('results-title').textContent = `Top ${data.recommendations.length} Recommendations`;
            document.querySelector('#results-section .section-subtitle').textContent = 
                `${gpuName}${count > 1 ? ' x' + count : ''} for "${useCase}"`;
        } else {
            showToast(data.error || 'No recommendations found', 'error');
            resultsSection.classList.add('hidden');
            showcaseSection.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Recommendation error:', error);
        showToast('Failed to get recommendations', 'error');
        resultsSection.classList.add('hidden');
        showcaseSection.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Get Recommendations';
    }
}

function updateGPUInfo() {
    const select = document.getElementById('gpu-select');
    const countInput = document.getElementById('gpu-count');
    const vramEl = document.getElementById('gpu-vram-info');
    const tierEl = document.getElementById('gpu-tier-info');
    
    const selectedOption = select.options[select.selectedIndex];
    const perGpuVram = parseInt(selectedOption?.dataset?.vram) || 0;
    const tier = selectedOption?.dataset?.tier || '-';
    const count = parseInt(countInput?.value) || 1;
    const totalVram = perGpuVram * count;
    
    if (vramEl) vramEl.textContent = `${totalVram} GB`;
    if (tierEl) tierEl.textContent = tier.charAt(0).toUpperCase() + tier.slice(1);
}

function handleChipClick(chipElement) {
    const textarea = document.getElementById('use-case-input');
    const hasCustomText = textarea.dataset.hasCustomText === 'true';
    
    chipElement.classList.toggle('active');
    
    const selectedValues = Array.from(document.querySelectorAll('#use-case-chips .chip.active')).map(c => c.dataset.value);
    
    if (selectedValues.length > 0) {
        textarea.value = selectedValues.join(' + ');
        textarea.dataset.hasCustomText = 'false';
    } else if (!hasCustomText) {
        textarea.value = '';
    }
}

function getSelectedUseCases() {
    const textarea = document.getElementById('use-case-input');
    const activeChips = document.querySelectorAll('#use-case-chips .chip.active');
    const selectedValues = Array.from(activeChips).map(c => c.dataset.value);
    
    if (selectedValues.length > 0) {
        return selectedValues.join(' + ');
    }
    
    return textarea.value.trim() || 'general';
}

function handleBackToShowcase() {
    const showcaseSection = document.getElementById('showcase-section');
    const resultsSection = document.getElementById('results-section');
    showcaseSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
    const gpuSelect = document.getElementById('gpu-select');
    if (gpuSelect) {
        gpuSelect.addEventListener('change', updateGPUInfo);
    }
    
    const gpuCount = document.getElementById('gpu-count');
    if (gpuCount) {
        gpuCount.addEventListener('input', updateGPUInfo);
    }

    const recommendBtn = document.getElementById('recommend-btn');
    if (recommendBtn) {
        recommendBtn.addEventListener('click', fetchRecommendations);
    }

    const chipsContainer = document.getElementById('use-case-chips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (chip) {
                handleChipClick(chip);
            }
        });
    }

    const useCaseInput = document.getElementById('use-case-input');
    if (useCaseInput) {
        useCaseInput.addEventListener('input', function() {
            const activeChips = document.querySelectorAll('#use-case-chips .chip.active');
            if (activeChips.length > 0) {
                this.dataset.hasCustomText = 'false';
            } else {
                this.dataset.hasCustomText = this.value.trim() ? 'true' : 'false';
            }
        });
    }

    const backBtn = document.getElementById('back-btn');
    if (backBtn) {
        backBtn.addEventListener('click', handleBackToShowcase);
    }

    showcaseCarousel = new Carousel('showcase-carousel', 'showcase-counter');
    showcaseCarousel.init();
    
    fetchShowcase();
});

window.addEventListener('resize', () => {
    showcaseCarousel?.updateCarousel();
    resultsCarousel?.updateCarousel();
});