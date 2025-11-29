"""
Tests for the Model builder class.
"""

import pytest
import numpy as np

from cuprox import Model
from cuprox.model import Variable, LinearExpr, Constraint


class TestVariable:
    """Tests for Variable class."""
    
    def test_variable_creation(self):
        """Test creating a variable."""
        model = Model()
        x = model.add_var(lb=0, ub=10, name="x")
        
        assert isinstance(x, Variable)
        assert x.lb == 0
        assert x.ub == 10
        assert x.name == "x"
        assert x.index == 0
    
    def test_variable_default_bounds(self):
        """Test default variable bounds."""
        model = Model()
        x = model.add_var()
        
        assert x.lb == 0
        assert x.ub == float('inf')
    
    def test_add_multiple_vars(self):
        """Test adding multiple variables."""
        model = Model()
        vars = model.add_vars(5, lb=0, ub=1, name_prefix="y")
        
        assert len(vars) == 5
        assert all(v.lb == 0 for v in vars)
        assert all(v.ub == 1 for v in vars)
        assert vars[0].name == "y_0"
        assert vars[4].name == "y_4"


class TestLinearExpr:
    """Tests for LinearExpr class."""
    
    def test_variable_addition(self):
        """Test adding variables."""
        model = Model()
        x = model.add_var(name="x")
        y = model.add_var(name="y")
        
        expr = x + y
        assert isinstance(expr, LinearExpr)
        assert expr.terms[x.index] == 1
        assert expr.terms[y.index] == 1
    
    def test_scalar_multiplication(self):
        """Test multiplying variable by scalar."""
        model = Model()
        x = model.add_var(name="x")
        
        expr = 3 * x
        assert isinstance(expr, LinearExpr)
        assert expr.terms[x.index] == 3
    
    def test_complex_expression(self):
        """Test complex expression building."""
        model = Model()
        x = model.add_var(name="x")
        y = model.add_var(name="y")
        
        expr = 2*x + 3*y - 5
        assert expr.terms[x.index] == 2
        assert expr.terms[y.index] == 3
        assert expr.constant == -5
    
    def test_negation(self):
        """Test negating a variable."""
        model = Model()
        x = model.add_var(name="x")
        
        expr = -x
        assert expr.terms[x.index] == -1


class TestConstraint:
    """Tests for Constraint class."""
    
    def test_le_constraint(self):
        """Test <= constraint."""
        model = Model()
        x = model.add_var(name="x")
        y = model.add_var(name="y")
        
        constr = x + y <= 10
        assert isinstance(constr, Constraint)
        assert constr.sense == "<="
    
    def test_ge_constraint(self):
        """Test >= constraint."""
        model = Model()
        x = model.add_var(name="x")
        
        constr = x >= 5
        assert isinstance(constr, Constraint)
        assert constr.sense == ">="
    
    def test_eq_constraint(self):
        """Test == constraint."""
        model = Model()
        x = model.add_var(name="x")
        y = model.add_var(name="y")
        
        constr = x + y == 10
        assert isinstance(constr, Constraint)
        assert constr.sense == "=="


class TestModel:
    """Tests for Model class."""
    
    def test_empty_model(self):
        """Test creating an empty model."""
        model = Model()
        assert model.num_vars == 0
        assert model.num_constrs == 0
    
    def test_add_constraint(self):
        """Test adding a constraint."""
        model = Model()
        x = model.add_var(name="x")
        y = model.add_var(name="y")
        
        model.add_constr(x + y <= 10, name="capacity")
        
        assert model.num_constrs == 1
    
    def test_minimize_objective(self):
        """Test setting minimize objective."""
        model = Model()
        x = model.add_var(name="x")
        y = model.add_var(name="y")
        
        model.minimize(-x - y)
        
        assert model._sense == "minimize"
        assert model._objective is not None
    
    def test_maximize_objective(self):
        """Test setting maximize objective."""
        model = Model()
        x = model.add_var(name="x")
        
        model.maximize(x)
        
        assert model._sense == "maximize"
    
    def test_to_standard_form(self):
        """Test conversion to standard form."""
        model = Model()
        x = model.add_var(lb=0, name="x")
        y = model.add_var(lb=0, name="y")
        
        model.add_constr(x + 2*y <= 10)
        model.add_constr(3*x + y <= 15)
        model.minimize(-x - y)
        
        A, b, c, lb, ub, senses = model._to_standard_form()
        
        assert c.shape == (2,)
        assert b.shape == (2,)
        assert lb.shape == (2,)
        assert ub.shape == (2,)
        
        np.testing.assert_array_equal(c, [-1, -1])
        np.testing.assert_array_equal(b, [10, 15])
    
    def test_model_repr(self):
        """Test model string representation."""
        model = Model()
        model.add_vars(5)
        model.add_constr(model._vars[0] <= 10)
        
        repr_str = repr(model)
        assert "vars=5" in repr_str
        assert "constrs=1" in repr_str

