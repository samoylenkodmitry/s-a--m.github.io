---
layout: leetcode-entry
title: "592. Fraction Addition and Subtraction"
permalink: "/leetcode/problem/2024-08-23-592-fraction-addition-and-subtraction/"
leetcode_ui: true
entry_slug: "2024-08-23-592-fraction-addition-and-subtraction"
---

[592. Fraction Addition and Subtraction](https://leetcode.com/problems/fraction-addition-and-subtraction/description/) easy
[blog post](https://leetcode.com/problems/fraction-addition-and-subtraction/solutions/5678060/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23082024-592-fraction-addition-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pycLsCkzhAQ)
![1.webp](/assets/leetcode_daily_images/9462af36.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/711

#### Problem TLDR

Eval string of fractions sum #medium #math

#### Intuition

The hardest part of this task is to remember how to simplify fractions like `12/18`. Both numbers' greatest common divisor is `6`, and the fraction is equivalent to `2/3`.

The GCD part also must be learned: `f(a,b)=a%b==0?b:f(b%a, a)`.

#### Approach

* we can parse numbers one by one, or we can parse symbol by symbol; the former is simpler to implement than the latter

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun fractionAddition(expression: String): String {
        var n1 = 0; var d1 = 1; var i = 0
        fun gcd(a: Int, b: Int): Int = if (a % b == 0) b else gcd(b % a, a)
        fun num() = expression.drop(i).takeWhile { it.isDigit() }
        while (i < expression.length) {
            var sign = 1
            if (expression[i] == '-') { sign = -1; i++ }
            if (expression[i] == '+') i++
            var n2 = sign * num().run { i += length + 1; toInt() }
            var d2 = num().run { i += length; toInt() }
            n1 = n1 * d2 + n2 * d1; d1 *= d2
            val gcd = gcd(abs(n1), d1)
            n1 /= gcd; d1 /= gcd
        }
        return "$n1/$d1"
    }

```
```rust

    pub fn fraction_addition(expression: String) -> String {
        let (mut n1, mut d1, mut d, mut p, mut sign) = (0, 1, 0, 0, 1);
        fn gcd(a: i32, b: i32) -> i32 { if a % b == 0 { b } else { gcd(b % a, a)}}
        for c in expression.bytes().chain([b'+'].into_iter()) { match c {
                b'0'..=b'9' => d = d * 10 + (c as u8 - b'0' as u8) as i32,
                b'/' => { p = sign * d; d = 0 },
                b'+' | b'-' => {
                    sign = if c == b'-' { -1 } else { 1 };
                    let n2 = p; let d2 = d.max(1);
                    n1 = n1 * d2 + n2 * d1; d1 *= d2;
                    let gcd = gcd(n1.abs(), d1);
                    n1 /= gcd; d1 /= gcd; d = 0
                },
                _ => {}
        }}; format!("{n1}/{d1}")
    }

```

