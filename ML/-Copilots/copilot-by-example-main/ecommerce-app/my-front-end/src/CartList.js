import React from 'react';
import CartListItem from './CartListItem';
import './CartList.css';

/**
 * Renders a list of products in the cart.
 *
 * @param {Object[]} products - The array of products in the cart.
 * @param {string} products[].id - The unique identifier of the product.
 * @param {string} products[].name - The name of the product.
 * @param {number} products[].price - The price of the product.
 * @returns {JSX.Element} The rendered cart list.
 */
const CartList = ({ products }) => {
  const handleProceed = () => {
    alert('Proceeding to checkout!');
  };

  return (
    <div className="cart-container">
      <h3 className="cart-title">Shopping Cart ({products.length} items)</h3>
      {products.map((product) => (
        <CartListItem key={product.id} name={product.name} price={product.price} />
      ))}
      <button className="proceed-button" onClick={handleProceed}>
        Proceed to Checkout
      </button>
    </div>
  );
};

export default CartList;
