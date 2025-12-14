const API_URL = 'http://localhost:8000';
const ANALYSIS_API_URL = 'http://localhost:8001';
let currentSubreddit = '';
let currentOffset = 0;
let isLoading = false;
let hasMorePosts = true;

const EMOTION_EMOJIS = {
    joy: '😊',
    neutral: '😐',
    anger: '😠',
    surprise: '😲',
    sadness: '😢',
    fear: '😨'
};

async function analyzeContent(posts, startOffset) {
    const textsToAnalyze = [];
    const textMap = new Map();
    
    console.log('Starting analysis with offset:', startOffset);
    console.log('Number of posts to analyze:', posts.length);
    
    posts.forEach((post, postIdx) => {
        if (post.text && post.text.trim()) {
            const id = `post-${startOffset + postIdx}`;
            console.log('Adding post for analysis:', id);
            textsToAnalyze.push(post.text);
            textMap.set(textsToAnalyze.length - 1, { id, text: post.text, type: 'post' });
        }
        
        function collectComments(replies, parentPath = []) {
            if (!replies) return;
            replies.forEach((reply, replyIdx) => {
                if (reply.text && reply.text.trim()) {
                    const path = [...parentPath, replyIdx];
                    const id = `comment-${startOffset + postIdx}-${path.join('-')}`;
                    console.log('Adding comment for analysis:', id);
                    textsToAnalyze.push(reply.text);
                    textMap.set(textsToAnalyze.length - 1, { id, text: reply.text, type: 'comment' });
                }
                if (reply.replies) {
                    collectComments(reply.replies, [...parentPath, replyIdx]);
                }
            });
        }
        
        if (post.replies) {
            collectComments(post.replies);
        }
    });
    
    if (textsToAnalyze.length === 0) return;
    
    console.log('Total texts to analyze:', textsToAnalyze.length);
    console.log('TextMap size:', textMap.size);
    
    try {
        const response = await fetch(`${ANALYSIS_API_URL}/api/predict/combined`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ texts: textsToAnalyze })
        });
        
        const data = await response.json();
        
        console.log('API Response:', data);
        console.log('Number of predictions received:', data.predictions?.length);
        console.log('TextMap size:', textMap.size);
        
        if (data.success && data.predictions) {
            // Only process predictions up to the number of texts we sent
            const numPredictions = Math.min(data.predictions.length, textMap.size);
            console.log('Processing predictions:', numPredictions);
            
            for (let idx = 0; idx < numPredictions; idx++) {
                const pred = data.predictions[idx];
                const textInfo = textMap.get(idx);
                if (!textInfo) {
                    console.log('No text info for index:', idx);
                    continue;
                }
                
                const element = document.getElementById(textInfo.id);
                if (!element) {
                    console.log('No element found for:', textInfo.id);
                    continue;
                }
                
                const analysisContainer = element.querySelector('.analysis-container');
                if (!analysisContainer) {
                    console.log('No analysis container for:', textInfo.id);
                    continue;
                }
                
                console.log('Rendering analysis for:', textInfo.id, pred);
                analysisContainer.innerHTML = renderAnalysis(pred, textInfo.id);
            }
            
            setTimeout(() => {
                document.querySelectorAll('.emotion-bar-fill').forEach(bar => {
                    const width = bar.getAttribute('data-width');
                    bar.style.width = width;
                });
            }, 100);
        }
    } catch (error) {
        console.error('Error analyzing content:', error);
        console.error('Error details:', error.message, error.stack);
        document.querySelectorAll('.analysis-container').forEach(container => {
            const analyzing = container.querySelector('.analyzing-badge');
            if (analyzing) {
                analyzing.textContent = '⚠️ Analysis unavailable';
                analyzing.style.color = '#e74c3c';
            }
        });
    }
}

function renderAnalysis(pred, elementId) {
    console.log('Rendering analysis:', pred);
    
    // Validate the prediction object
    if (!pred || !pred.toxicity || !pred.emotion) {
        console.error('Invalid prediction structure:', pred);
        return '<div class="analyzing-badge" style="color: #e74c3c;">⚠️ Invalid data structure</div>';
    }
    
    const toxicity = pred.toxicity;
    const emotion = pred.emotion;
    const distribution = emotion.distribution;
    
    if (!distribution) {
        console.error('No distribution found:', emotion);
        return '<div class="analyzing-badge" style="color: #e74c3c;">⚠️ Missing emotion distribution</div>';
    }
    
    const sortedEmotions = Object.entries(distribution)
        .sort((a, b) => b[1] - a[1]);
    
    const toxClass = toxicity.label === 'toxic' ? 'toxicity-toxic' : 'toxicity-non-toxic';
    
    return `
        <div class="analysis-section">
            <div class="analysis-header">
                <div class="metric-badge ${toxClass}">
                    ${toxicity.label}
                    <span class="confidence-text">${(toxicity.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-badge emotion-primary">
                    ${EMOTION_EMOJIS[emotion.label]} ${emotion.label}
                    <span class="confidence-text">${(emotion.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
            
            <button class="expand-button" onclick="toggleDistribution('${elementId}')">
                📊 Show Emotion Distribution
            </button>
            
            <div id="dist-${elementId}" class="distribution-container distribution-collapsed">
                <div class="distribution-title">
                    📊 Emotion Distribution
                </div>
                ${sortedEmotions.map(([emotionName, value]) => {
                    const percentage = (value * 100).toFixed(1);
                    return `
                        <div class="emotion-bar">
                            <div class="emotion-bar-header">
                                <div class="emotion-label">
                                    <span>${EMOTION_EMOJIS[emotionName]}</span>
                                    <span>${emotionName}</span>
                                </div>
                                <div class="emotion-percentage">${percentage}%</div>
                            </div>
                            <div class="emotion-bar-bg">
                                <div class="emotion-bar-fill emotion-${emotionName}" 
                                        data-width="${percentage}%" 
                                        style="width: 0%">
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;
}

function toggleDistribution(elementId) {
    const distElement = document.getElementById(`dist-${elementId}`);
    const button = event.target;
    
    if (distElement.classList.contains('distribution-collapsed')) {
        distElement.classList.remove('distribution-collapsed');
        distElement.classList.add('distribution-expanded');
        button.textContent = '📊 Hide Emotion Distribution';
    } else {
        distElement.classList.remove('distribution-expanded');
        distElement.classList.add('distribution-collapsed');
        button.textContent = '📊 Show Emotion Distribution';
    }
}

async function fetchData(isNewSearch = false) {
    const subreddit = document.getElementById('subredditInput').value.trim();
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const statsDiv = document.getElementById('stats');
    const searchBtn = document.getElementById('searchBtn');
    const loadMoreDiv = document.getElementById('loadMore');
    const endMessageDiv = document.getElementById('endMessage');

    if (!subreddit) {
        showError('Please enter a subreddit name');
        return;
    }

    if (isLoading) return;
    isLoading = true;

    if (isNewSearch) {
        currentSubreddit = subreddit;
        currentOffset = 0;
        hasMorePosts = true;
        resultsDiv.innerHTML = '';
        loadMoreDiv.style.display = 'none';
        endMessageDiv.style.display = 'none';
    }

    errorDiv.style.display = 'none';
    loadingDiv.style.display = 'block';
    searchBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/api/subreddit/${subreddit}?max_posts=10&offset=${currentOffset}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch data');
        }

        displayResults(data, isNewSearch);
        
        if (data.posts && data.posts.length > 0) {
            // Capture offset AFTER displayResults, which updates currentOffset
            // But we need the offset BEFORE it was updated
            const offsetForAnalysis = currentOffset - data.posts.length;
            // Use setTimeout to ensure DOM is fully updated before analyzing
            setTimeout(async () => {
                await analyzeContent(data.posts, offsetForAnalysis);
            }, 100);
            loadMoreDiv.style.display = 'block';
        } else if (currentOffset === 0) {
            resultsDiv.innerHTML = '<div class="no-results">No posts found for this subreddit.</div>';
            hasMorePosts = false;
        } else {
            hasMorePosts = false;
            loadMoreDiv.style.display = 'none';
            endMessageDiv.style.display = 'block';
        }

    } catch (error) {
        showError(error.message);
        hasMorePosts = false;
    } finally {
        loadingDiv.style.display = 'none';
        searchBtn.disabled = false;
        isLoading = false;
    }
}

async function loadMorePosts() {
    if (!hasMorePosts || isLoading) return;
    
    isLoading = true;
    const loadMoreBtn = document.querySelector('#loadMore button');
    const loadingMoreDiv = document.getElementById('loadingMore');
    const loadMoreDiv = document.getElementById('loadMore');
    const endMessageDiv = document.getElementById('endMessage');
    
    loadMoreBtn.disabled = true;
    loadMoreDiv.style.display = 'none';
    loadingMoreDiv.style.display = 'block';

    try {
        const response = await fetch(`${API_URL}/api/subreddit/${currentSubreddit}?max_posts=20&offset=${currentOffset}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch more posts');
        }

        if (data.posts && data.posts.length > 0) {
            const offsetForAnalysis = currentOffset - data.posts.length;
            displayResults(data, false);
            // Use setTimeout to ensure DOM is fully updated
            setTimeout(async () => {
                await analyzeContent(data.posts, offsetForAnalysis);
            }, 100);
            loadMoreDiv.style.display = 'block';
        } else {
            hasMorePosts = false;
            endMessageDiv.style.display = 'block';
        }

    } catch (error) {
        showError('Failed to load more posts: ' + error.message);
        hasMorePosts = false;
    } finally {
        loadingMoreDiv.style.display = 'none';
        loadMoreBtn.disabled = false;
        isLoading = false;
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function displayResults(data, isNewSearch) {
    const resultsDiv = document.getElementById('results');
    const statsDiv = document.getElementById('stats');

    if (!data.posts || data.posts.length === 0) {
        if (isNewSearch) {
            resultsDiv.innerHTML = '<div class="no-results">No posts found for this subreddit.</div>';
        }
        return;
    }

    const newPostCount = data.posts.length;
    const totalPosts = currentOffset + newPostCount;

    statsDiv.innerHTML = `<strong>r/${data.subreddit}</strong> - ${totalPosts} posts loaded`;
    statsDiv.style.display = 'block';

    const postsHtml = data.posts.map((post, idx) => {
        const postId = `post-${currentOffset + idx}`;
        return `
            <div class="post" id="${postId}">
                <div class="post-header">
                    <div class="post-title">${escapeHtml(post.topic || 'No title')}</div>
                    <div class="post-meta">
                        Posted by <span class="author">u/${escapeHtml(post.author)}</span>
                    </div>
                </div>
                ${post.text ? `
                    <div class="post-text">${escapeHtml(post.text)}</div>
                    <div class="analysis-container">
                        <div class="analyzing-badge">⏳ Analyzing...</div>
                    </div>
                ` : ''}
                ${post.replies && post.replies.length > 0 ? renderReplies(post.replies, currentOffset + idx) : ''}
            </div>
        `;
    }).join('');

    if (isNewSearch) {
        resultsDiv.innerHTML = postsHtml;
    } else {
        resultsDiv.innerHTML += postsHtml;
    }

    currentOffset = totalPosts;
}

function renderReplies(replies, postIdx, parentPath = [], isNested = false) {
    if (!replies || replies.length === 0) return '';

    const containerClass = isNested ? 'nested-reply' : 'replies';
    const header = !isNested ? '<div class="replies-header">Comments:</div>' : '';

    return `
        <div class="${containerClass}">
            ${header}
            ${replies.map((reply, idx) => {
                const path = [...parentPath, idx];
                const commentId = `comment-${postIdx}-${path.join('-')}`;
                return `
                    <div class="comment" id="${commentId}">
                        <div class="comment-author">u/${escapeHtml(reply.author)}</div>
                        <div class="comment-text">${escapeHtml(reply.text || '')}</div>
                        ${reply.text ? `
                            <div class="analysis-container">
                                <div class="analyzing-badge">⏳ Analyzing...</div>
                            </div>
                        ` : ''}
                        ${reply.replies && reply.replies.length > 0 ? renderReplies(reply.replies, postIdx, path, true) : ''}
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

document.addEventListener('DOMContentLoaded', function() {
    const searchBtn = document.getElementById('searchBtn');
    const subredditInput = document.getElementById('subredditInput');
    
    if (searchBtn) {
        searchBtn.addEventListener('click', () => fetchData(true));
    }
    
    if (subredditInput) {
        subredditInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                fetchData(true);
            }
        });
    }
});