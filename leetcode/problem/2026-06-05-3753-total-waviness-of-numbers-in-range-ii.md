---
layout: leetcode-entry
title: "3753. Total Waviness of Numbers in Range II"
permalink: "/leetcode/problem/2026-06-05-3753-total-waviness-of-numbers-in-range-ii/"
leetcode_ui: true
entry_slug: "2026-06-05-3753-total-waviness-of-numbers-in-range-ii"
---

[3753. Total Waviness of Numbers in Range II](https://leetcode.com/problems/total-waviness-of-numbers-in-range-ii/solutions/8315031/kotlin-rust-by-samoylenkodmitry-tvg7/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05062026-3753-total-waviness-of-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/G5IQBvlhUsw)

https://dmitrysamoylenko.com/leetcode/

![05.06.2026.webp](/assets/leetcode_daily_images/05.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1381

#### Problem TLDR

Count hills and valleys base10 in a..b

#### Intuition

Didn't solve myself.
Brute-force - would not work for 10^15 elements.
Digit DP: try every digit, be aware of the limit and of the leading zeros.

#### Approach

* when counting valleys or hills we must add entire suffix of possible values: they will be either limited (suffix of number itself) or not (pow10)

#### Complexity

- Time complexity:
$$O(logn)$$

- Space complexity:
$$O(logn)$$

#### Code

```kotlin
    fun totalWaviness(a: Long, b: Long): Long {
        val P = (1..16).scan(1L) { r, _ -> r * 10L }
        fun c(x: Long): Long {
            val s = "$x"; val dp = HashMap<String, Long>()
            fun f(i: Int, pp: Int, p: Int, t: Boolean): Long =
                if (i == s.length) 0L else dp.getOrPut("$i $pp $p $t") {
                    (0..if (t) 9 else s[i] - '0').sumOf { d ->
                        val nt = t || d < s[i] - '0'
                        val w = if (pp >= 0 && (pp - p) * (d - p) > 0)
                            if (nt) P[s.length - 1 - i] else (s.substring(i + 1).toLongOrNull() ?: 0L) + 1L else 0L
                        w + f(i + 1, if (p < 0 && d == 0) -1 else p, if (p < 0 && d == 0) -1 else d, nt)
                    }
                }
            return f(0, -1, -1, false)
        }
        return c(b) - c(a - 1)
    }
```
```rust
    pub fn total_waviness(a: i64, b: i64) -> i64 {
        fn c(x: i64) -> i64 {
            fn f(i: usize, u: i8, v: i8, t: bool, x: i64, s: &[u8], m: &mut HashMap<(usize, i8, i8, bool), i64>) -> i64 {
                if let Some(&r) = m.get(&(i, u, v, t)) { return r; }
                let (mut r, j, l) = (0, (s.len() - 1 - i) as u32, if t { 9 } else { (s[i] - 48) as i8 });
                for d in 0..=l {
                    let (n, z, p) = (t || d < l, v < 0 && d == 0, 10i64.pow(j));
                    if u >= 0 && (u - v) * (d - v) > 0 { r += if n { p } else { x % p + 1 }; }
                    if j > 0 { r += f(i + 1, if z { -1 } else { v }, if z { -1 } else { d }, n, x, s, m); }
                }
                m.insert((i, u, v, t), r); r
            }
            if x < 0 { 0 } else { f(0, -1, -1, false, x, &x.to_string().into_bytes(), &mut HashMap::new()) }
        }
        c(b) - c(a - 1)
    }
```

