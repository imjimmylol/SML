import numpy as np

class RunningMeanStd:
    """
    Tracks running mean and standard deviation using Welford's online algorithm.
    This is numerically stable and commonly used in RL frameworks like OpenAI Baselines.
    """
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Args:
            epsilon: Small constant for numerical stability when normalizing
            shape: Shape of the data (e.g., () for scalar, (4,) for 4D vector)
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Start with small count to avoid division by zero
    
    def update(self, x):
        """
        Update statistics with new batch of data.
        
        Args:
            x: New data, shape can be (shape,) or (batch_size, *shape)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.mean.shape) else 1
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """
        Update from precomputed batch statistics (useful for distributed training).
        Uses parallel algorithm for combining statistics.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x, clip_range=None):
        """
        Normalize data using current statistics.
        
        Args:
            x: Data to normalize
            clip_range: If provided, clip normalized values to [-clip_range, clip_range]
        
        Returns:
            Normalized data
        """
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        if clip_range is not None:
            normalized = np.clip(normalized, -clip_range, clip_range)
        return normalized
