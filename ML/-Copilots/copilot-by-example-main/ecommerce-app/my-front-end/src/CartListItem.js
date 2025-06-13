import React from 'react';
import './CartListItem.css';

const CartListItem = ({ name, price }) => {
  return (
    <div className="cart-list-item">
      <h3 className="cart-item-name">{name}</h3>
      <p className="cart-item-price">${price}</p>
    </div>
  );
};

export default CartListItem;
