---
layout: leetcode-entry
title: "1963. Minimum Number of Swaps to Make the String Balanced"
permalink: "/leetcode/problem/2024-10-08-1963-minimum-number-of-swaps-to-make-the-string-balanced/"
leetcode_ui: true
entry_slug: "2024-10-08-1963-minimum-number-of-swaps-to-make-the-string-balanced"
---

[1963. Minimum Number of Swaps to Make the String Balanced](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/solutions/5885284/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08102024-1963-minimum-number-of-swaps?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8HKy3tumvk0)
![1.webp](/assets/leetcode_daily_images/9fe3fe41.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/761

#### Problem TLDR

Min swaps to balance brackets #medium #two_pointers #stack

#### Intuition

Let's observe how we can do the balancing:

```j

    // 012345
    // ][][][
    // i    j
    //  i  j

    // 012345
    // ]]][[[
    // i    j
    // [i  j]

```
One way is to go with two pointers `i` from the begining and `j` from the end. Skip all balanced brackets and swap non-balanced positions.

Another thought is to go with a stack and do the 'swap' on unbalanced position by making it balanced.

#### Approach

* to virtually swap, change the closing bracket `c = -1` to opening `c = 1`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minSwaps(s: String): Int {
        var balance = 0; var res = 0
        for (c in s) if (c == '[') balance++
            else if (balance == 0) { res++; balance = 1 }
            else balance--
        return res
    }

```
```rust

    pub fn min_swaps(s: String) -> i32 {
        let (mut c, mut res) = (0, 0);
        for b in s.bytes() { if b == b'[' { c += 1 }
            else if c == 0 { res += 1; c = 1 }
            else { c -= 1 }}
        res
    }

```
```c++

    int minSwaps(string s) {
        int b = 0, res = 0;
        for (char c: s) if (c == '[') b++;
            else if (b == 0) { res++; b = 1; }
            else b--;
        return res;
    }

```

