---
layout: leetcode-entry
title: "3713. Longest Balanced Substring I"
permalink: "/leetcode/problem/2026-02-12-3713-longest-balanced-substring-i/"
leetcode_ui: true
entry_slug: "2026-02-12-3713-longest-balanced-substring-i"
---

[3713. Longest Balanced Substring I](https://leetcode.com/problems/longest-balanced-substring-i/description) medium
[blog post](https://leetcode.com/problems/longest-balanced-substring-i/solutions/7573416/kotlin-rust-by-samoylenkodmitry-ds63/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12022026-3713-longest-balanced-substring?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tyAS-4wTnBc)

![8720b147-110d-40c5-83e4-c98559eb564f (1).webp](/assets/leetcode_daily_images/b32916bf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1266

#### Problem TLDR

Max substring with equal frequencies #medium #sliding_window

#### Intuition

Brute-force.
1. Start from every index. Go to the end. Compute the frequencies, Detect when all f the same.
2. Check every window size decreasing. Compute the frequencies in a sliding window. Stop when all f the same.

#### Approach

* max(f) * uniq = window

#### Complexity

- Time complexity:
$$O(26n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 123ms
    fun longestBalanced(s: String) = (s.length downTo 1).first { w ->
        val f = IntArray(26); var u = 0
        s.indices.any { i ->
            if (f[s[i]-'a']++ < 1) ++u
            if (i - w >= 0) if (--f[s[i-w]-'a'] < 1) --u
            1 + i - w >= 0 && w == f.max() * u
        }
    }
```
```rust
// 16ms
    pub fn longest_balanced(s: String) -> i32 {
        let b = s.as_bytes(); (0..b.len()).map(|i| {
            let (mut f, mut u, mut m) = ([0; 26], 0, 0);
            b[i..].iter().enumerate().map(|(len, &c)| {
                let k = (c - b'a') as usize;
                if f[k] == 0 { u += 1 }; f[k] += 1; m = m.max(f[k]);
                if m * u == len + 1 { (len + 1) as i32 } else { 0 }
            }).max().unwrap_or(0)
        }).max().unwrap_or(0)
    }
```

