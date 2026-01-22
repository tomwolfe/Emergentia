"""
Transformer-based Symbolic Expression Generator to move towards basis-free symbolic discovery.
This addresses the reliance on hand-crafted basis functions in the FeatureTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random


class SymbolicVocabulary:
    """
    Vocabulary for symbolic expressions containing operators, variables, and constants.
    """
    
    def __init__(self):
        # Special tokens
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Mathematical operators
        self.operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs']
        
        # Variables (will be indexed based on problem dimension)
        self.variables = []  # Will be populated based on problem
        
        # Constants
        self.constants = ['0', '1', '2', 'pi', 'e']
        
        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        for token in special_tokens:
            self._add_token(token)
        
        # Add operators
        for op in self.operators:
            self._add_token(op)
        
        # Add constants
        for const in self.constants:
            self._add_token(const)
    
    def _add_token(self, token):
        """Add a token to the vocabulary."""
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def add_variables(self, n_vars):
        """Add variable tokens x0, x1, ..., x{n_vars-1}."""
        for i in range(n_vars):
            var_name = f"x{i}"
            self._add_token(var_name)
            self.variables.append(var_name)
    
    def encode_sequence(self, tokens):
        """Encode a sequence of tokens to IDs."""
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]
    
    def decode_sequence(self, ids):
        """Decode a sequence of IDs to tokens."""
        return [self.id_to_token.get(idx, self.unk_token) for idx in ids]
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to understand sequence order.
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SymbolicTransformer(nn.Module):
    """
    Transformer model for generating symbolic expressions.
    """
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, max_seq_length=100, dropout=0.1):
        super(SymbolicTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for the transformer.
        
        Args:
            src: Source sequence [batch_size, seq_len] (token IDs)
            src_mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        src_emb = self.embedding(src) * np.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        
        # Pass through transformer
        output = self.transformer_encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Generate output logits
        output = self.fc_out(output)
        
        return output
    
    def generate(self, start_token_id, max_length=50, temperature=1.0, device='cpu'):
        """
        Generate a symbolic expression sequence autoregressively.
        
        Args:
            start_token_id: ID of the start token
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (lower = more deterministic)
            device: Device to run generation on
            
        Returns:
            Generated sequence of token IDs
        """
        self.eval()
        
        with torch.no_grad():
            # Start with the start token
            generated = torch.full((1, 1), start_token_id, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Get model predictions
                output = self.forward(generated)
                
                # Get the prediction for the last token
                next_token_logits = output[0, -1, :] / temperature
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample the next token
                next_token_id = torch.multinomial(probs, 1).item()
                
                # Append to generated sequence
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                generated = torch.cat([generated, next_token_tensor], dim=1)
                
                # Stop if end token is generated
                if next_token_id == 2:  # Assuming end_token_id is 2
                    break
            
            return generated.squeeze().tolist()


class SymbolicExpressionGenerator:
    """
    Main class for generating symbolic expressions using the transformer model.
    """
    
    def __init__(self, n_variables=4, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, max_seq_length=100):
        self.vocab = SymbolicVocabulary()
        self.vocab.add_variables(n_variables)
        
        self.model = SymbolicTransformer(
            vocab_size=self.vocab.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length
        )
        
        self.n_variables = n_variables
        self.max_seq_length = max_seq_length
    
    def train(self, dataloader, optimizer, criterion, num_epochs=100, device='cpu'):
        """
        Train the transformer on symbolic expression data.
        
        Args:
            dataloader: DataLoader with symbolic expression sequences
            optimizer: Optimizer for training
            criterion: Loss function (typically CrossEntropyLoss)
            num_epochs: Number of training epochs
            device: Device to train on
        """
        self.model.to(device)
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_src, batch_tgt in dataloader:
                batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch_src)
                
                # Reshape for loss calculation
                output = output.view(-1, self.vocab.vocab_size)
                tgt = batch_tgt.view(-1)
                
                # Calculate loss (ignore padding tokens)
                loss = criterion(output, tgt)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    def generate_expression(self, temperature=0.8, max_length=30):
        """
        Generate a new symbolic expression.
        
        Args:
            temperature: Sampling temperature
            max_length: Maximum length of generated expression
            
        Returns:
            String representation of the generated expression
        """
        import re
        device = next(self.model.parameters()).device
        
        # Helper for diverse fallback
        def get_fallback():
            # Return a random linear combination of 2 random variables
            v1, v2 = random.sample(range(self.n_variables), 2)
            c1, c2 = random.uniform(0.01, 0.2), random.uniform(0.01, 0.2)
            return f"{c1:.2f} * x{v1} + {c2:.2f} * x{v2}"

        # Get start token ID
        start_token_id = self.vocab.token_to_id[self.vocab.start_token]
        
        # Generate sequence
        generated_ids = self.model.generate(
            start_token_id=start_token_id,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        
        # Decode to tokens
        tokens = self.vocab.decode_sequence(generated_ids)
        
        # Filter out special tokens using regex
        cleaned_tokens = [t for t in tokens if not re.match(r'<.*?>', t)]
        
        # Convert to expression string
        expression_str = ' '.join(cleaned_tokens)
        
        # Validation Step
        if not expression_str.strip():
            return get_fallback()
            
        try:
            # Map common tokens to SymPy equivalents for validation
            local_dict = {
                'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log,
                'sqrt': sp.sqrt, 'abs': sp.Abs, 'pi': sp.pi, 'e': sp.E
            }
            for i in range(self.n_variables):
                local_dict[f'x{i}'] = sp.Symbol(f'x{i}')
            
            # Use sympify to validate if it's valid math
            sp.sympify(expression_str, locals=local_dict)
            return expression_str
        except Exception:
            # If not valid math, return diverse fallback
            return get_fallback()
    
    def parse_to_sympy(self, expression_str):
        """
        Attempt to parse the generated expression string to SymPy.
        
        Args:
            expression_str: Generated expression string
            
        Returns:
            SymPy expression or None if parsing fails
        """
        try:
            # Simple parsing for basic expressions
            # This is a simplified version - a full parser would be more complex
            tokens = expression_str.split()
            
            # Map common tokens to SymPy equivalents
            local_dict = {
                'sin': sp.sin,
                'cos': sp.cos,
                'exp': sp.exp,
                'log': sp.log,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
                'pi': sp.pi,
                'e': sp.E
            }
            
            # Add variables
            for i in range(self.n_variables):
                local_dict[f'x{i}'] = sp.Symbol(f'x{i}')
            
            # Convert to infix notation if needed
            # For now, assuming prefix notation from transformer
            expr_str = ' '.join(tokens)
            
            # Try to parse using SymPy
            parsed_expr = sp.sympify(expr_str, locals=local_dict)
            return parsed_expr
            
        except Exception as e:
            print(f"Failed to parse expression '{expression_str}': {e}")
            return None
    
    def sample_expressions(self, n_samples=5, temperature=0.8):
        """
        Generate multiple symbolic expressions.
        
        Args:
            n_samples: Number of expressions to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated expressions
        """
        expressions = []
        for _ in range(n_samples):
            expr = self.generate_expression(temperature=temperature)
            expressions.append(expr)
        return expressions


class NeuralSymbolicHybrid:
    """
    Hybrid system that combines neural networks with the transformer-based symbolic generator.
    """
    
    def __init__(self, neural_model, symbolic_generator, blend_factor=0.5):
        """
        Args:
            neural_model: The trained neural model (e.g., DiscoveryEngineModel)
            symbolic_generator: SymbolicExpressionGenerator instance
            blend_factor: Factor controlling the blend between neural and symbolic (0 to 1)
        """
        self.neural_model = neural_model
        self.symbolic_generator = symbolic_generator
        self.blend_factor = blend_factor
    
    def generate_candidates(self, n_candidates=10, temperature=0.8):
        """
        Generate candidate symbolic expressions.
        
        Args:
            n_candidates: Number of candidate expressions to generate
            temperature: Sampling temperature
            
        Returns:
            List of candidate expressions
        """
        return self.symbolic_generator.sample_expressions(
            n_samples=n_candidates, 
            temperature=temperature
        )
    
    def evaluate_candidates(self, candidates, data_loader, metric='mse'):
        """
        Evaluate generated candidates against neural model predictions.
        
        Args:
            candidates: List of candidate expressions
            data_loader: DataLoader with validation data
            metric: Evaluation metric ('mse', 'mae', etc.)
            
        Returns:
            List of (candidate, score) tuples sorted by score
        """
        evaluations = []
        
        for candidate in candidates:
            try:
                # Parse the candidate expression
                sympy_expr = self.symbolic_generator.parse_to_sympy(candidate)
                if sympy_expr is None:
                    evaluations.append((candidate, float('inf')))
                    continue
                
                # Evaluate against neural model predictions
                total_error = 0
                n_samples = 0
                
                for batch in data_loader:
                    # Get neural model predictions
                    with torch.no_grad():
                        neural_pred = self.neural_model(batch.x, batch.edge_index, batch.batch)
                    
                    # TODO: Implement comparison between symbolic and neural predictions
                    # This would require evaluating the symbolic expression on the same inputs
                    # For now, we'll just assign a dummy score
                    error = 0.0  # Placeholder
                    
                    total_error += error
                    n_samples += 1
                
                avg_error = total_error / max(n_samples, 1)
                evaluations.append((candidate, avg_error))
                
            except Exception:
                evaluations.append((candidate, float('inf')))
        
        # Sort by score (lower is better)
        evaluations.sort(key=lambda x: x[1])
        return evaluations
    
    def distill_best_candidate(self, data_loader, n_candidates=20, temperature=0.8):
        """
        Distill the best symbolic expression from generated candidates.
        
        Args:
            data_loader: DataLoader with validation data
            n_candidates: Number of candidates to generate and evaluate
            temperature: Sampling temperature
            
        Returns:
            Best symbolic expression
        """
        candidates = self.generate_candidates(n_candidates, temperature)
        evaluations = self.evaluate_candidates(candidates, data_loader)
        
        # Return the best candidate
        if evaluations:
            best_candidate, best_score = evaluations[0]
            return best_candidate, best_score
        else:
            return None, float('inf')


def create_neural_symbolic_hybrid(neural_model, n_variables=4, blend_factor=0.5):
    """
    Factory function to create a neural-symbolic hybrid system.
    
    Args:
        neural_model: Trained neural model
        n_variables: Number of input variables for symbolic expressions
        blend_factor: Blend factor between neural and symbolic
        
    Returns:
        NeuralSymbolicHybrid instance
    """
    generator = SymbolicExpressionGenerator(n_variables=n_variables)
    return NeuralSymbolicHybrid(neural_model, generator, blend_factor)