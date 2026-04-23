---
layout: leetcode-entry
title: "838. Push Dominoes"
permalink: "/leetcode/problem/2025-05-02-838-push-dominoes/"
leetcode_ui: true
entry_slug: "2025-05-02-838-push-dominoes"
---

[838. Push Dominoes](https://leetcode.com/problems/push-dominoes/description/) medium
[blog post](https://leetcode.com/problems/push-dominoes/solutions/6707005/kotlin-rust-by-samoylenkodmitry-svrs/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02052025-838-push-dominoes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yq8GhGFsvYM)
![1.webp](/assets/leetcode_daily_images/b9223286.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/976

#### Problem TLDR

Dominoes simulation #medium

#### Intuition

My first idea was to use force balance, but I didn't find the working algorigthm for that (it is possible to make it work, but force should be decreasing from n)
```j
    // 0123456789
    // .L.R...LR..L..
    // 00012340123000 to the right
    // 21004321032100 to the left
    //    1?02 1?
    // LL.RR.LLRR
    // 21  20?1 1
```
The more simple approach is to notice how `..L`, `R..`, `L..R` and `R..L` ranges are behaving. Then it is all about the implementation details.

#### Approach

* the golfed code here is completely obfuscates the logic: consider only `L` and backtrack to the half of `R` range or full otherwise.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

// 30ms
    fun pushDominoes(d: String) = buildString {
        var r = false; var l = 1
        for ((i, c) in d.withIndex())
            if (c == '.') { append(if (r) 'R' else c); ++l } else { append(c)
                if (c == 'L') for (j in (if (r) i - l / 2 else i - l + 1)..<i) set(j, c)
                if (c == 'L' && r && l % 2 < 1) set(i - l / 2, '.')
                r = c == 'R'; l = 1
            }
    }

```
```kotlin

// 10ms
    fun pushDominoes(dominoes: String): String {
        val r = dominoes.toCharArray(); var isR = false; var l = 1
        for ((i, c) in r.withIndex())
            if (c == '.') { if (isR) r[i] = 'R'; ++l } else {
                if (c == 'L') for (j in (if (isR) i - l / 2 else i - l + 1)..<i) r[j] = 'L'
                if (c == 'L' && isR && l % 2 < 1) r[i - l / 2] = '.'
                isR = c == 'R'; l = 1
            }
        return String(r)
    }

```
```rust

// 1ms
    pub fn push_dominoes(mut d: String) -> String {
        unsafe { let (mut r, mut l, mut b) = (false, 1, d.as_bytes_mut());
        for i in 0..b.len() {
            if b[i] == b'.' { if r { b[i] = b'R'}; l += 1 } else {
                if b[i] == b'L' { b[if r { i - l / 2 } else { i - l + 1 }..i].fill(b'L') }
                if b[i] == b'L' && r && l % 2 < 1 { b[i - l / 2] = b'.' }
                r = b[i] == b'R'; l = 1
            }}} d
    }

```
```c++

// 0ms
    string pushDominoes(string d) {
        for (int i = 0, n = size(d), l = 1, r = 0; i < n; ++i)
            if (d[i] == '.') { if (r) d[i] = 'R'; ++l; } else {
                if (d[i] == 'L') fill(begin(d) + (r ? i - l/2 : i - l + 1), begin(d) + i, 'L');
                if (d[i] == 'L' && r && l % 2 < 1) d[i - l/2] = '.';
                r = d[i] == 'R'; l = 1;
            }
        return d;
    }

```

