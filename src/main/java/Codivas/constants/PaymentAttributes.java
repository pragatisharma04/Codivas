package Codivas.constants;

import lombok.Setter;

@Setter
public class PaymentAttributes {
    String type, customerName, destinationName;
    Integer amount, oldCustomerBalance, newCustomerBalance, oldDestinationBalance,newDestinationBalance;
}
