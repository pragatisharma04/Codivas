package Codivas.constants;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PaymentAttributes {
    String type, customerName, destinationName;
    float amount, oldCustomerBalance, newCustomerBalance, oldDestinationBalance,newDestinationBalance;
}
