---
layout: leetcode-entry
title: "564. Find the Closest Palindrome"
permalink: "/leetcode/problem/2024-08-24-564-find-the-closest-palindrome/"
leetcode_ui: true
entry_slug: "2024-08-24-564-find-the-closest-palindrome"
---

[564. Find the Closest Palindrome](https://leetcode.com/problems/find-the-closest-palindrome/description/) hard
[blog post](https://leetcode.com/problems/find-the-closest-palindrome/solutions/5683025/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24082024-564-find-the-closest-palindrome?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KEopNyW4e_U)
![1.webp](/assets/leetcode_daily_images/c77e4f0c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/712

#### Problem TLDR

Closest palindrome number #hard #math

#### Intuition

Let's observe the possible results for some examples:

```j
    // 54321 543
    // 54345
    // 54
    // 55
    // 12345
    // 12321

    // 12321
    // 12221

    // 12021
    // 11911
    // 12121

    // 101
    // 99
    // 111

    // 1001
    // 999
    // 1111

    // 1000001
    // 1001001
    // 999999

    // 2000002
    // 1999991
    // 2001002

    // 11
    // 1001
    //  9

    // 1551
    // 1441
```
As we see, there are not too many of them: we should consider the left half, then increment or decrement it.
There are too many corner cases, however and this is the main hardness of this problem.

#### Approach

* Let's just try `9`-nth, and `101`-th as a separate candidates.
* For odd case, we should avoid to double the middle

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun nearestPalindromic(n: String): String {
        val half = n.take(n.length / 2 + (n.length % 2))
        val a = half + half.reversed().drop(n.length % 2)
        var b = "${half.toInt() - 1}"; b += "$b".reversed().drop(n.length % 2)
        var c = "${half.toInt() + 1}"; c += "$c".reversed().drop(n.length % 2)
        val d = "0${"9".repeat(n.length - 1)}"
        val e = "1${"0".repeat(n.length - 1)}1"
        return listOf(a, b, c, d, e).filter { it != n }.map { it.toLong() }
            .minWith(compareBy({ abs(it - n.toLong() )}, { it })).toString()
    }

```
```rust

    pub fn nearest_palindromic(n: String) -> String {
        let (len, n) = (n.len() as u32, n.parse::<i64>().unwrap());
        (-1..2).map(|i| {
            let mut h = (n / 10i64.pow(len / 2) + i).to_string();
            let mut r: String = h.chars().rev().skip(len as usize % 2).collect();
            (h + &r).parse().unwrap()
        }).chain([10i64.pow(len - 1) - 1, 10i64.pow(len) + 1 ])
        .filter(|&x| x != n)
        .min_by_key(|&x| ((x - n).abs(), x)).unwrap().to_string()
    }

```

