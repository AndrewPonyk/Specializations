import React from 'react';
import CartListItem from './CartListItem';

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
  return (
    <div>
      {products.map((product) => (
        <CartListItem key={product.id} name={product.name} price={product.price} />
      ))}
    </div>
  );
};

export default CartList;
