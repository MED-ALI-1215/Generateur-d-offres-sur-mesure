/**
 * ==============================================================================
 * CYBERSECURITY AI OFFER GENERATOR - COMPLETE FRONTEND
 * ==============================================================================
 */

class OfferGenerator {
    constructor() {
        this.currentGeneration = null;
        this.isGenerating = false;
        this.healthData = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkServerHealth();
        this.loadTaxonomy();
        console.log('🚀 Offer Generator initialized');
    }

    bindEvents() {
        // Main buttons
        document.getElementById('generateBtn')?.addEventListener('click', () => this.generateOffer());
        document.getElementById('clearBtn')?.addEventListener('click', () => this.clearInput());
        document.getElementById('newOfferBtn')?.addEventListener('click', () => this.resetToInput());

        // Action buttons
        document.getElementById('copyBtn')?.addEventListener('click', () => this.copyOffer());
        document.getElementById('downloadBtn')?.addEventListener('click', () => this.downloadOffer());
        document.getElementById('classifyBtn')?.addEventListener('click', () => this.classifyQuestion());
        document.getElementById('systemStatusBtn')?.addEventListener('click', () => this.showSystemStatus());

        // Example cards
        document.querySelectorAll('.example-card').forEach(card => {
            card.addEventListener('click', () => this.useExample(card.dataset.example));
        });

        // Input validation
        const input = document.getElementById('questionInput');
        if (input) {
            input.addEventListener('input', () => {
                this.validateInput();
                this.debounceClassification(input.value);
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }

    async checkServerHealth() {
        try {
            const response = await fetch('/health');
            this.healthData = await response.json();
            console.log('Server health:', this.healthData);
        } catch (error) {
            console.error('Health check failed:', error);
            this.showError('Server connection error');
        }
    }

    async loadTaxonomy() {
        try {
            const response = await fetch('/taxonomy');
            this.taxonomy = await response.json();
            console.log('Taxonomy loaded:', this.taxonomy);
        } catch (error) {
            console.error('Failed to load taxonomy:', error);
        }
    }

    validateInput() {
        const input = document.getElementById('questionInput');
        const generateBtn = document.getElementById('generateBtn');
        if (!input || !generateBtn) return;

        const isValid = input.value.trim().length >= 10;
        generateBtn.disabled = !isValid || this.isGenerating;
        
        // Visual feedback
        if (input.value.trim().length === 0) {
            input.style.borderColor = '#dee2e6';
        } else if (isValid) {
            input.style.borderColor = '#28a745';
        } else {
            input.style.borderColor = '#ffc107';
        }
    }

    debounceClassification(value) {
        clearTimeout(this.classificationTimeout);
        this.classificationTimeout = setTimeout(() => {
            if (value.trim().length > 5) {
                this.previewClassification(value);
            }
        }, 1000);
    }

    async previewClassification(question) {
        try {
            const response = await fetch('/taxonomy/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            const result = await response.json();
            this.showClassificationResult(result);
        } catch (error) {
            console.error('Classification preview failed:', error);
        }
    }

    async generateOffer() {
        if (this.isGenerating) return;

        const question = document.getElementById('questionInput').value.trim();
        if (!question) {
            this.showError('Please describe your cybersecurity needs');
            return;
        }

        this.isGenerating = true;
        this.showLoadingSection();
        this.disableGenerateButton();

        try {
            console.log('🔄 Starting offer generation...');
            
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText);
            }
            
            const result = await response.json();
            console.log('📋 Generation result:', result);
            
            this.currentGeneration = result;
            this.showResults(result);
            this.showToast('Offer generated successfully!', 'success');
            
        } catch (error) {
            console.error('Generation error:', error);
            this.showError(error.message || 'Generation failed');
        } finally {
            this.isGenerating = false;
            this.enableGenerateButton();
            this.hideLoadingSection();
        }
    }

    showLoadingSection() {
        const loadingSection = document.getElementById('loadingSection');
        const resultsSection = document.getElementById('resultsSection');
        
        if (loadingSection) loadingSection.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';
        
        this.animateProgressBar();
    }

    animateProgressBar() {
        const progressBar = document.getElementById('progressBar');
        if (!progressBar) return;

        const steps = [
            { id: 'step-retrieve', progress: 30, delay: 500 },
            { id: 'step-grade_documents', progress: 50, delay: 1500 },
            { id: 'step-generate', progress: 80, delay: 2500 },
            { id: 'step-validate', progress: 100, delay: 3500 }
        ];

        steps.forEach(({ id, progress, delay }) => {
            setTimeout(() => {
                if (this.isGenerating) {
                    progressBar.style.width = `${progress}%`;
                    this.activateStep(id);
                }
            }, delay);
        });
    }

    activateStep(stepId) {
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'completed');
        });

        const currentStep = document.getElementById(stepId);
        if (currentStep) {
            currentStep.classList.add('active');
        }
    }

    hideLoadingSection() {
        const loadingSection = document.getElementById('loadingSection');
        if (loadingSection) {
            loadingSection.style.display = 'none';
        }
        
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'completed');
        });
        
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = '0%';
        }
    }

    showResults(result) {
        console.log('📊 Showing results:', result);
        
        this.hideLoadingSection();
        
        // Debug metrics structure
        console.log('🔍 Debug metrics structure:', JSON.stringify(result.metrics, null, 2));
        console.log('🔍 Quality metrics:', JSON.stringify(result.metrics?.quality, null, 2));
        
        // Update metrics dashboard
        this.updateMetrics(result.metrics);
        
        // Display generated offer
        const offerContent = document.getElementById('generatedOffer');
        if (offerContent) {
            if (result.final_result) {
                offerContent.innerHTML = this.formatOffer(result.final_result);
            } else {
                offerContent.innerHTML = '<p class="text-muted">No offer generated</p>';
            }
        }

        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    updateMetrics(metrics) {
        if (!metrics) {
            console.warn('❌ No metrics provided');
            return;
        }

        console.log('📊 Updating metrics:', metrics);

        // Core performance metrics
        this.updateMetricElement('processingTime', `${metrics.processing_time || 0}s`);
        this.updateMetricElement('documentsCount', `${metrics.relevant_documents || 0}/${metrics.documents_retrieved || 0}`);
        this.updateMetricElement('confidenceScore', `${metrics.confidence_score || 0}%`, 
            this.getColorForScore((metrics.confidence_score || 0) / 100));
        
        // Memory usage
        this.updateMetricElement('memoryUsage', `${metrics.memory_usage || 0} MB`);
        
        // Pipeline efficiency
        this.updateMetricElement('pipelineEfficiency', `${metrics.pipeline_efficiency || 0}%`);
        
        // Taxonomy classification
        const primaryCategory = metrics.taxonomy?.primary || 'None';
        const secondaryCategories = metrics.taxonomy?.secondary?.join(', ') || '';
        const categoryText = primaryCategory === 'None' ? 'None' : 
            `${primaryCategory}${secondaryCategories ? ` (${secondaryCategories})` : ''}`;
        this.updateMetricElement('categoryDetected', categoryText);

        // Quality metrics - FIXED to handle correct structure
        if (metrics.quality) {
            console.log('📈 Quality metrics found:', metrics.quality);
            
            this.updateMetricElement('qualityGrade', metrics.quality.grade || 'F', 
                `grade-${(metrics.quality.grade || 'f').toLowerCase()}`);
            
            // Use detailed_scores if available
            if (metrics.quality.detailed_scores) {
                this.updateMetricElement('precisionScore', 
                    `${Math.round((metrics.quality.detailed_scores.precision || 0) * 100)}%`, 
                    this.getColorForScore(metrics.quality.detailed_scores.precision || 0));
                
                this.updateMetricElement('consistencyScore', 
                    `${Math.round((metrics.quality.detailed_scores.consistency || 0) * 100)}%`, 
                    this.getColorForScore(metrics.quality.detailed_scores.consistency || 0));
                
                this.updateMetricElement('overallScore', 
                    `${Math.round((metrics.quality.overall_score || 0) * 100)}%`, 
                    this.getColorForScore(metrics.quality.overall_score || 0));
                
                // Completeness if element exists
                this.updateMetricElement('completenessScore', 
                    `${Math.round((metrics.quality.detailed_scores.completeness || 0) * 100)}%`, 
                    this.getColorForScore(metrics.quality.detailed_scores.completeness || 0));
            } else {
                console.warn('❌ No detailed_scores in quality metrics');
            }
        } else {
            console.warn('❌ No quality metrics found, setting defaults');
            // Set default values for quality metrics
            this.updateMetricElement('qualityGrade', 'N/A');
            this.updateMetricElement('precisionScore', '0%');
            this.updateMetricElement('consistencyScore', '0%');
            this.updateMetricElement('overallScore', '0%');
            this.updateMetricElement('completenessScore', '0%');
        }

        // Cost metrics
        this.updateMetricElement('estimatedCost', `$${(metrics.cost || 0).toFixed(4)}`);
        
        // Token usage (if elements exist)
        this.updateMetricElement('inputTokens', `${metrics.input_tokens || 0}`);
        this.updateMetricElement('outputTokens', `${metrics.output_tokens || 0}`);
        this.updateMetricElement('totalTokens', `${(metrics.input_tokens || 0) + (metrics.output_tokens || 0)}`);

        console.log('✅ Metrics update completed');
    }

    updateMetricElement(id, value, className = '') {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            if (className) {
                // Handle color codes
                if (className.startsWith('#')) {
                    element.style.color = className;
                } else {
                    element.className = className;
                }
            }
            console.log(`📊 Updated ${id}: ${value}`);
        } else {
            console.warn(`⚠️ Element with id '${id}' not found in DOM`);
        }
    }

    getColorForScore(score) {
        if (score === null || score === undefined || isNaN(score)) return '#6c757d'; // Gray for no data
        if (score >= 0.8) return '#28a745'; // Green
        if (score >= 0.6) return '#ffc107'; // Yellow
        if (score >= 0.4) return '#fd7e14'; // Orange
        return '#dc3545'; // Red
    }

    formatOffer(offerText) {
        if (!offerText) return '<p class="text-muted">No offer generated</p>';
        
        // Basic formatting
        let formatted = offerText
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Section headers (French and English)
        const sections = [
            'RÉSUMÉ EXÉCUTIF', 'EXECUTIVE SUMMARY',
            'SOLUTION TECHNIQUE', 'TECHNICAL SOLUTION',
            'PLAN DE MISE EN ŒUVRE', 'IMPLEMENTATION PLAN',
            'CALENDRIER', 'TIMELINE', 'SCHEDULE',
            'COMPOSITION DE L\'ÉQUIPE', 'TEAM COMPOSITION',
            'RÉSULTATS ATTENDUS', 'EXPECTED RESULTS',
            'STRUCTURE DE PRIX', 'PRICING STRUCTURE', 'COST STRUCTURE'
        ];

        sections.forEach(section => {
            const regex = new RegExp(`(${section.replace(/'/g, "\\'")}:?)`, 'gi');
            formatted = formatted.replace(regex, '<h3 class="section-header">$1</h3>');
        });

        return `<div class="offer-content">${formatted}</div>`;
    }

    async classifyQuestion() {
        const question = document.getElementById('questionInput').value.trim();
        if (!question) {
            this.showError('Please enter a question to classify');
            return;
        }

        try {
            const response = await fetch('/taxonomy/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            
            if (!response.ok) throw new Error('Classification failed');
            
            const result = await response.json();
            this.showClassificationResult(result);
            this.showToast('Question classified successfully!');
        } catch (error) {
            console.error('Classification error:', error);
            this.showError('Classification failed');
        }
    }

    showClassificationResult(result) {
        const preview = document.getElementById('classificationPreview');
        if (!preview) return;

        if (result.classification?.primary) {
            const confidence = Math.round((result.classification.confidence || 0) * 100);
            const secondary = result.classification.secondary?.length ? 
                ` (${result.classification.secondary.join(', ')})` : '';
            
            preview.innerHTML = `
                <i class="fas fa-tag"></i> 
                Detected: <strong>${result.classification.primary}</strong>${secondary}
                <span class="confidence-badge">${confidence}% confidence</span>
            `;
            preview.style.display = 'block';
        } else {
            preview.style.display = 'none';
        }
    }

    copyOffer() {
        if (!this.currentGeneration?.final_result) {
            this.showError('No offer to copy');
            return;
        }
        
        navigator.clipboard.writeText(this.currentGeneration.final_result)
            .then(() => this.showToast('Copied to clipboard!', 'success'))
            .catch(() => this.showError('Copy failed'));
    }

    downloadOffer() {
        if (!this.currentGeneration?.final_result) {
            this.showError('No offer to download');
            return;
        }
        
        const blob = new Blob([this.currentGeneration.final_result], { type: 'text/plain; charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cybersecurity-offer-${new Date().toISOString().slice(0,10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('Offer downloaded!', 'success');
    }

    resetToInput() {
        const resultsSection = document.getElementById('resultsSection');
        const questionInput = document.getElementById('questionInput');
        
        if (resultsSection) resultsSection.style.display = 'none';
        if (questionInput) questionInput.focus();
        
        this.currentGeneration = null;
        this.showToast('Ready for new offer generation');
    }

    useExample(exampleText) {
        const input = document.getElementById('questionInput');
        if (input) {
            input.value = exampleText;
            input.focus();
            this.validateInput();
            this.showToast('Example loaded');
        }
    }

    clearInput() {
        const input = document.getElementById('questionInput');
        const generateBtn = document.getElementById('generateBtn');
        const classificationPreview = document.getElementById('classificationPreview');
        
        if (input) {
            input.value = '';
            input.style.borderColor = '#dee2e6';
        }
        
        if (generateBtn) generateBtn.disabled = true;
        if (classificationPreview) classificationPreview.style.display = 'none';
    }

    disableGenerateButton() {
        const btn = document.getElementById('generateBtn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        }
    }

    enableGenerateButton() {
        const btn = document.getElementById('generateBtn');
        const input = document.getElementById('questionInput');
        
        if (btn) {
            const isValid = input && input.value.trim().length >= 10;
            btn.disabled = !isValid;
            btn.innerHTML = '<i class="fas fa-magic"></i> Generate Offer';
        }
    }

    async showSystemStatus() {
        try {
            const response = await fetch('/test-components');
            if (!response.ok) throw new Error('Failed to get system status');
            
            const data = await response.json();
            this.showStatusModal(data.results);
        } catch (error) {
            console.error('System status error:', error);
            this.showError('Failed to get system status');
        }
    }

    showStatusModal(results) {
        let html = '<div class="status-grid">';
        
        for (const [component, status] of Object.entries(results)) {
            const statusClass = status.status === 'working' ? 'success' : 
                               status.status === 'error' ? 'error' : 'warning';
            const statusIcon = status.status === 'working' ? '✓' : '✗';
            
            html += `
                <div class="status-item">
                    <span class="status-icon ${statusClass}">${statusIcon}</span>
                    <div class="status-details">
                        <strong>${component}:</strong> ${status.status}
                        ${status.error ? `<br><small class="error-text">${status.error}</small>` : ''}
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        this.showModal('System Status', html);
    }

    showModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close" aria-label="Close">&times;</button>
                </div>
                <div class="modal-body">${content}</div>
                <div class="modal-footer">
                    <button class="btn btn-primary modal-close-btn">Close</button>
                </div>
            </div>
        `;
        
        // Close handlers
        modal.querySelector('.modal-close').addEventListener('click', () => this.closeModal(modal));
        modal.querySelector('.modal-close-btn').addEventListener('click', () => this.closeModal(modal));
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeModal(modal);
        });
        
        document.body.appendChild(modal);
    }

    closeModal(modal) {
        if (modal && modal.parentNode) {
            modal.parentNode.removeChild(modal);
        }
    }

    showToast(message, type = 'success', duration = 3000) {
        const toast = document.getElementById('toast');
        if (!toast) {
            console.log(`Toast: ${message}`);
            return;
        }

        toast.textContent = message;
        toast.className = `toast show toast-${type}`;
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    }

    showError(message) {
        console.error('Error:', message);
        
        const errorModal = document.getElementById('errorModal');
        if (errorModal) {
            const errorMessage = document.getElementById('errorMessage');
            if (errorMessage) errorMessage.textContent = message;
            errorModal.classList.add('show');
        } else {
            // Fallback to toast
            this.showToast(message, 'error', 5000);
        }
    }

    handleKeyboardShortcuts(e) {
        // Ctrl+Enter to generate
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (!this.isGenerating) {
                this.generateOffer();
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal-overlay').forEach(modal => this.closeModal(modal));
            const errorModal = document.getElementById('errorModal');
            if (errorModal && errorModal.classList.contains('show')) {
                errorModal.classList.remove('show');
            }
        }
        
        // Ctrl+K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const input = document.getElementById('questionInput');
            if (input) input.focus();
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Offer Generator...');
    window.offerGenerator = new OfferGenerator();
    
    // Global error handler
    window.addEventListener('error', (e) => {
        console.error('Global error:', e.error);
        if (window.offerGenerator) {
            window.offerGenerator.showError('An unexpected error occurred');
        }
    });
    
    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (e) => {
        console.error('Unhandled promise rejection:', e.reason);
        if (window.offerGenerator) {
            window.offerGenerator.showError('An unexpected error occurred');
        }
    });
    
    console.log('✅ Application initialized successfully');
});