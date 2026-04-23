---
layout: leetcode-entry
title: "1945. Sum of Digits of String After Convert"
permalink: "/leetcode/problem/2024-09-03-1945-sum-of-digits-of-string-after-convert/"
leetcode_ui: true
entry_slug: "2024-09-03-1945-sum-of-digits-of-string-after-convert"
---

[1945. Sum of Digits of String After Convert](https://leetcode.com/problems/sum-of-digits-of-string-after-convert/description/) easy
[blog post](https://leetcode.com/problems/sum-of-digits-of-string-after-convert/solutions/5729684/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03092024-1945-sum-of-digits-of-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NVh63AsZek8)
![1.webp](/assets/leetcode_daily_images/cd4b51cb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/723

#### Problem TLDR

Sum of number chars `k` times #easy #simulation

#### Intuition

* the first transformation is different: `c - 'a' + 1`
* other transformations: `c - '0'`

#### Approach

* we can do it with strings or with just numbers

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun getLucky(s: String, k: Int) = (1..<k).fold(
        s.map { "${it - 'a' + 1}" }.joinToString("").sumOf { it - '0' }
    ) { r, t -> r.toString().sumOf { it.code - '0'.code }}

```
```rust

    pub fn get_lucky(s: String, k: i32) -> i32 {
        let dig = |x| { let (mut s, mut x) = (0, x);
            while x > 0 { s += x % 10; x /= 10 }; s};
        (1..k).fold(s.bytes().map(|b|
            dig(b as i32 - 96)).sum(), |r, t| dig(r))
    }

```
```c++

    int getLucky(string s, int k) {
        auto dig = [](int x) {
            int s = 0;
            while (x > 0) { s += x % 10; x /= 10; }
            return s;
        };
        int sum = 0;
        for (char c : s) sum += dig(c - 'a' + 1);
        while (k-- > 1) sum = dig(sum);
        return sum;
    }

```

