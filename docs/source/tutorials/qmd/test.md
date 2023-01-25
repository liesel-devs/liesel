
# Quarto

Quarto enables you to weave together content and executable code into a
finished document. To learn more about Quarto see <https://quarto.org>.

``` python
import liesel.model as lsl
```

## Running Code

When you click the **Render** button a document will be generated that
includes both content and the output of embedded code. You can embed
code like this:

``` r
1 + 1
```

    [1] 2

You can add options to executable code like this

    [1] 4

The `echo: false` option disables the printing of code (only output is
displayed).

Equations:

$$
y_i = \beta_0 + x_i \beta_1 + \varepsilon_i
$$

``` r
x <- rnorm(10)
y <- rnorm(10)
plot(x, y)
```

![](test_files/figure-commonmark/unnamed-chunk-4-1.png)

``` r
x
```

     [1]  0.2340879 -0.5121327 -1.6835836  0.5842578 -0.8420712  0.7382087
     [7]  0.9664081  0.2391690  0.1099716 -0.7138898
