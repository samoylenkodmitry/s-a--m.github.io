---
layout: leetcode-entry
title: "2264. Largest 3-Same-Digit Number in String"
permalink: "/leetcode/problem/2025-08-14-2264-largest-3-same-digit-number-in-string/"
leetcode_ui: true
entry_slug: "2025-08-14-2264-largest-3-same-digit-number-in-string"
---

[2264. Largest 3-Same-Digit Number in String](https://leetcode.com/problems/largest-3-same-digit-number-in-string/description/) easy
[blog post](https://leetcode.com/problems/largest-3-same-digit-number-in-string/solutions/7078939/kotlin-rust-by-samoylenkodmitry-c3cp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14082025-2264-largest-3-same-digit?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GFkElrtuSB0)
![1.webp](/assets/leetcode_daily_images/e93b4397.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1080

#### Problem TLDR

Max 3-digit in string #easy

#### Intuition

Sliding window is accepted and the optimal

#### Approach

* compare just a single char
* let's explore the variations: what if we check each digit individually?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 26ms
    fun largestGoodInteger(num: String) = num.windowed(3)
    .filter { it.toSet().size < 2 }.maxByOrNull { it[0] } ?: ""

```
```kotlin

// 15ms
    fun largestGoodInteger(n: String) = (9 downTo 0)
        .map { "$it$it$it" }.find { it in n } ?: ""

```

```rust

// 0ms
    pub fn largest_good_integer(num: String) -> String {
        num.as_bytes().windows(3).filter(|w| w[0] == w[1] && w[1] == w[2])
        .max_by_key(|w| w[0]).map_or("".into(), |w| from_utf8(w).unwrap().into())
    }

```
```c++

// 0ms
    string largestGoodInteger(string n) {
        int c = 0, p = 0; char m = 0;
        for (auto x: n) c = p == x ? ++c : 1, p = x, m = c > 2 ? max(m, x):m;
        return m > 0 ? string()+m+m+m : "";
    }

```
```python

// 0ms
    def largestGoodInteger(_, n):
        return next((d*3 for d in '9876543210' if d*3 in n), '')

```

