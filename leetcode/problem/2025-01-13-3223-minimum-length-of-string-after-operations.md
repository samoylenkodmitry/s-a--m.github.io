---
layout: leetcode-entry
title: "3223. Minimum Length of String After Operations"
permalink: "/leetcode/problem/2025-01-13-3223-minimum-length-of-string-after-operations/"
leetcode_ui: true
entry_slug: "2025-01-13-3223-minimum-length-of-string-after-operations"
---

[3223. Minimum Length of String After Operations](https://leetcode.com/problems/minimum-length-of-string-after-operations/description/) medium
[blog post](https://leetcode.com/problems/minimum-length-of-string-after-operations/solutions/6272913/kotlin-rust-by-samoylenkodmitry-28pq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13012025-3223-minimum-length-of-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2wGjJb-DyMI)
[deep-dive](https://notebooklm.google.com/notebook/a16db30f-3778-418e-8734-6700f4eb66ac/audio)
![1.webp](/assets/leetcode_daily_images/a4811766.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/863

#### Problem TLDR

Length after removing repeatings > 2 #medium

#### Intuition

The takeaways are always either `1` or `2`:
```j

    // 1 -> 1
    // 2 -> 2
    // 3 -> 1
    // 4 -> 2
    // 5 -> 3 -> 1
    // 6 -> 4 -> 2
    // 7 -> 1
    // 8 -> 2

```

Count each char's frequency.

#### Approach

* we can do some arithmetics `2 - 2 * f`
* careful: only count existing characters
* we can apply bitmasks instead of `f[26]`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minimumLength(s: String) =
        s.groupBy { it }.values.sumBy {
            2 - it.size % 2
        }

```
```rust

    pub fn minimum_length(s: String) -> i32 {
        let mut f = vec![0; 26];
        for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        (0..26).filter(|&b| f[b] > 0).map(|b| 2 - f[b] % 2).sum()
    }

```
```c++

    int minimumLength(string s) {
        int e = 0, f = 0;
        for (char c: s)
            f ^= 1 << (c - 'a'), e |= 1 << (c - 'a');
        return 2 * __builtin_popcount(e) - __builtin_popcount(f & e);
    }

```

