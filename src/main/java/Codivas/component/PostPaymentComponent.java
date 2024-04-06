package Codivas.component;

import Codivas.constants.PaymentAttributes;
import Codivas.dao.DBAccessor;

public class PostPaymentComponent {
    DBAccessor dbAccessor = new DBAccessor();
    public String postPayment(PaymentAttributes paymentAttributes) {
        System.out.println("Component:");
        return dbAccessor.postPayment(paymentAttributes);
    }
}
