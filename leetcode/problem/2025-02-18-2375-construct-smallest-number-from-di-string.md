---
layout: leetcode-entry
title: "2375. Construct Smallest Number From DI String"
permalink: "/leetcode/problem/2025-02-18-2375-construct-smallest-number-from-di-string/"
leetcode_ui: true
entry_slug: "2025-02-18-2375-construct-smallest-number-from-di-string"
---

[2375. Construct Smallest Number From DI String](https://leetcode.com/problems/construct-smallest-number-from-di-string/description/) medium
[blog post](https://leetcode.com/problems/construct-smallest-number-from-di-string/solutions/6436756/kotlin-rust-by-samoylenkodmitry-gpos/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18022025-2375-construct-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/sxUflu52fDk)
![1.webp](/assets/leetcode_daily_images/aa267e61.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/899

#### Problem TLDR

Smallest number from 1..9 and Inc-Dec pattern #medium #backtrack #greedy

#### Intuition

The problem size is `7`, brute-force works: try every number, filter out by pattern.

The clever solution from u/votrubac/ (didn't find it myself) is greedy: we have a set of `123456789` and we skip `III`part, flip the `DDDD` part greedily. It works on-line by appending the final `I`:

```j

    // DDDIIID.
    // j     .
    // 1234  .
    // 4321  .
    //     j .
    //     5 .
    //      j.
    //      6.
    //       j
    //       7
    //       87

```

#### Approach

* the numbers are uniq
* use the bitmask
* the actual number of possible solutions is small: `IID -> 1243 2354 3465 4576 5687 6798, IIIIDDD -> 12348765, 2345876, 2345987`

#### Complexity

- Time complexity:
$$O(n^n)$$ brute-force, O(n!) with filters, O(n) for greedy

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun smallestNumber(p: String): String {
        fun dfs(i: Int, n: Int, m: Int): Int? =
            if (i > p.length) n else (1..9).firstNotNullOfOrNull { x ->
            if (1 shl x and m > 0 || (i > 0 && (p[i - 1] > 'D') != (x > n % 10)))
            null else dfs(i + 1, n * 10 + x, 1 shl x or m) }
        return "${dfs(0, 0, 0)}"
    }

```
```rust

    pub fn smallest_number(p: String) -> String {
        let (mut r, mut j) = (vec![], 0);
        for i in 0..=p.len() {
            r.push(b'1' + i as u8);
            if i == p.len() || p.as_bytes()[i] == b'I' {
                r[j..].reverse(); j = i + 1 } }; String::from_utf8(r).unwrap()
    }

```
```c++

    string smallestNumber(string p) {
        string r;
        for (int i = 0, j = 0; i <= size(p); ++i) {
            r += '1' + i;
            if (i == size(p) || p[i] > 'D') reverse(begin(r) + j, end(r)), j = i + 1;
        } return r;
    }

```

