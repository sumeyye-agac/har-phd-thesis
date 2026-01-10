"""
Knowledge distillation model (student + teacher + optional attention transfer).
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D


class Distiller(keras.Model):
    def __init__(self, student, teacher, architecture):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.architecture = architecture

        self.conv2d_layer_deepconvlstm = Conv2D(filters=4, kernel_size=(1, 1))
        # SqueezeNet support (for future use)
        # self.conv2d_layer_squeezenet = Conv2D(filters=6, kernel_size=(1, 1))

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=1,
        beta=0.0,
        attention_list=None,
        attention_layer=None,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta
        self.attention_list = attention_list or []
        self.attention_layer = attention_layer

    def call(self, x, training=False):
        return self.student(x, training=training)

    def train_step(self, data):
        x, y = data

        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) * (self.temperature**2)

            if "SP_ATT" in self.attention_list:
                teacher_attention = self.teacher.get_layer(
                    f"spatial_attention_{self.attention_layer}"
                ).get_attention()
                student_attention = self.student.get_layer(
                    f"spatial_attention_{self.attention_layer}"
                ).get_attention()

                attention_loss = tf.reduce_mean(tf.square(teacher_attention - student_attention))
                loss = (
                    self.alpha * student_loss
                    + (1 - self.alpha) * distillation_loss
                    + self.beta * attention_loss
                )

            elif "CH_ATT" in self.attention_list and self.architecture == "deepconvlstm":
                teacher_attention = self.teacher.get_layer(
                    f"channel_attention_{self.attention_layer}"
                ).get_attention()
                student_attention = self.student.get_layer(
                    f"channel_attention_{self.attention_layer}"
                ).get_attention()

                teacher_attention1 = self.conv2d_layer_deepconvlstm(teacher_attention)
                attention_loss = tf.reduce_mean(tf.square(teacher_attention1 - student_attention))
                loss = (
                    self.alpha * student_loss
                    + (1 - self.alpha) * distillation_loss
                    + self.beta * attention_loss
                )

            # SqueezeNet support (for future use)
            # elif "CH_ATT" in self.attention_list and self.architecture == "squeezenet":
            #     teacher_attention = self.teacher.get_layer(
            #         f"channel_attention_{self.attention_layer}"
            #     ).get_attention()
            #     student_attention = self.student.get_layer(
            #         f"channel_attention_{self.attention_layer}"
            #     ).get_attention()
            #
            #     if self.attention_layer == 1:
            #         teacher_attention1 = self.conv2d_layer_squeezenet(teacher_attention)
            #         attention_loss = tf.reduce_mean(tf.square(teacher_attention1 - student_attention))
            #     else:
            #         attention_loss = tf.reduce_mean(tf.square(teacher_attention - student_attention))
            #
            #     loss = (
            #         self.alpha * student_loss
            #         + (1 - self.alpha) * distillation_loss
            #         + self.beta * attention_loss
            #     )

            else:
                attention_loss = 0.0
                loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {
                "student_loss": student_loss,
                "distillation_loss": distillation_loss,
                "attention_loss": attention_loss,
            }
        )
        return results

    def test_step(self, data):
        x, y = data

        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)

        self.compiled_metrics.update_state(y, y_prediction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
