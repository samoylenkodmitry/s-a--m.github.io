---
layout: leetcode-entry
title: "166. Fraction to Recurring Decimal"
permalink: "/leetcode/problem/2025-09-24-166-fraction-to-recurring-decimal/"
leetcode_ui: true
entry_slug: "2025-09-24-166-fraction-to-recurring-decimal"
---

[166. Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/description) medium
[blog post](https://leetcode.com/problems/fraction-to-recurring-decimal/solutions/7219619/kotlin-rust-by-samoylenkodmitry-95of/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24092025-166-fraction-to-recurring?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/inwxxh-viLE)

![1.webp](/assets/leetcode_daily_images/f983dbba.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1122

#### Problem TLDR

Calc a/b to string x.y(z) #medium #math

#### Intuition

Gave up to implement this correctly.

```j
    // how to know it is repeating?
    // let's just brute-force 10^4 digits
    // ok, how to find repeats? kmp (i forgot it)?
    // just store visited "(a/b)"
    // ok 43 minute, looks for hints, any simpler ideas? (no)
    // decide to give up, no time for debugging this

```
The ideas:
* to divide 1/3 multiply 1*10, and repeat the problem for 1%3 / 3
* to find the repeating part remember the problem "1/3" or just "1"
* to find where the repeating part start, remember the positions for each key "1"
* solve the part before "." before going next

#### Approach

* abs(Int.MIN_VALUE) == Int.MIN_VALUE, convert to longs before `abs`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 2ms
    fun fractionToDecimal(n: Int, d: Int) = buildString {
        if (1L * n.sign * d.sign < 0) append("-")
        var n = abs(1L*n); val d = abs(1L*d)
        append(n / d); n %= d; if (n > 0L) append(".")
        val visited = HashMap<Long, Int>()
        while (n > 0L) {
            n *= 10; append(n / d); n %= d
            visited.put(n, length)?.let {
                insert(it, "("); append(")"); n = 0L
            }
        }
    }

```

```rust

// 0ms
    pub fn fraction_to_decimal(n: i32, d: i32) -> String {
        let (n, d, mut s) = (n as i64, d as i64, String::new());
        if n * d < 0 { s.push('-') }; let (n, d) = (n.abs(), d.abs());
        s.push_str(&(n/d).to_string()); let mut n = n % d; if n > 0 {s.push('.')};
        let mut m = HashMap::new();
        while n > 0 {
            if let Some(&i) = m.get(&n) { s.insert(i, '('); s.push(')'); break }
            m.insert(n, s.len()); n *= 10; s.push_str(&(n/d).to_string()); n %= d
        } s
    }

```

