---
layout: leetcode-entry
title: "1318. Minimum Flips to Make a OR b Equal to c"
permalink: "/leetcode/problem/2023-06-07-1318-minimum-flips-to-make-a-or-b-equal-to-c/"
leetcode_ui: true
entry_slug: "2023-06-07-1318-minimum-flips-to-make-a-or-b-equal-to-c"
---

[1318. Minimum Flips to Make a OR b Equal to c](https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/description/) medium
[blog post](https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/solutions/3607170/kotlin-or-and-xor/)
[substack](https://dmitriisamoilenko.substack.com/p/07062023-1318-minimum-flips-to-make?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/238
#### Problem TLDR
Minimum `a` and `b` Int bit flips to make `a or b == c`.
#### Intuition
Naive implementation is to iterate over `32` bits and flip `a` or/and `b` bits to match `c`.
If we didn't consider the case where `a = 1` and `b = 1` and `c = 0`, the result would be `(a or b) xor c`, as `a or b` gives us the left side of the equation, and `xor c` gives only bits that are needed to flip. For the corner case `a = b = 1, c = 0`, we must do additional flip to make `0`, and we must make any other combinations `0`:

```

a b c     a and b   c.inv()   (a and b) and c.inv()

0 0 1     0         0         0
0 1 0     0         1         0
0 1 1     0         0         0
1 0 0     0         1         0
1 0 1     0         0         0
1 1 0     1         1         1
1 1 1     1         0         0

```

#### Approach
Use `Integer.bitCount`.

#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun minFlips(a: Int, b: Int, c: Int): Int =
Integer.bitCount((a or b) xor c) + Integer.bitCount((a and b) and c.inv())

```

