---
layout: leetcode-entry
title: "1569. Number of Ways to Reorder Array to Get Same BST"
permalink: "/leetcode/problem/2026-10-18-1569-number-of-ways-to-reorder-array-to-get-same-bst/"
leetcode_ui: true
entry_slug: "2026-10-18-1569-number-of-ways-to-reorder-array-to-get-same-bst"
---

[1569. Number of Ways to Reorder Array to Get Same BST](https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/solutions/8577786/kotlin-rust-by-samoylenkodmitry-aqw8/) hard
[substack](https://dmitriisamoilenko.substack.com/p/18102026-1569-number-of-ways-to-reorder?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/fm36K2FIETk)

https://dmitrysamoylenko.com/leetcode/

![18.10.2026.webp](/assets/leetcode_daily_images/18.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1515

#### Problem TLDR

Ways to permute keeping the same BST

#### Intuition

Didn't solve.
Facts:
1. order of the items in the left branch stays the same; order on the right stays the same
2. to merge left and right subtrees we picking either first from the left or first from the right, this is a dp choice problem c(l,r)=c(l-1,r)+c(l,r-1)
3. or it could be viewed as combinatorics problem of 'place left count on left+right vacant places' (or place the right count, doesn't matter)

#### Approach

* to write it without the recursion, we have to expand the math of merge: C (L+R L) * C (L + R L) = C (left_size+right_size left_size) across every node = N!/Product(sizes of all subtrees)
* sizes of subtrees can be known in a single loop: next_large-prev_smaller-1 is the range
* there is a monotonic stack solution exists (solves the same problem of knowing the subtree size)

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun numOfWays(n: IntArray) = run {
        val m = 1000000007.toBigInteger(); val s = TreeSet(setOf(0, n.size + 1))
        var f = 1.toBigInteger(); var d = 1.toBigInteger()
        n.forEachIndexed { i, x -> s += x
            f = f * (i + 1).toBigInteger() % m
            d = d * (s.higher(x)!! - s.lower(x)!! - 1).toBigInteger() % m
        }
        ((f * d.modInverse(m) + m - 1.toBigInteger()) % m).toInt()
    }
```
```rust
    pub fn num_of_ways(n: Vec<i32>) -> i32 {
        const M: i64 = 1_000_000_007; let mut c = [[1; 1001]; 1001];
        for i in 2..=n.len() { for j in 1..i {c[i][j]=(c[i-1][j-1]+c[i-1][j])%M}}
        fn f(a: &[i32], c: &[[i64; 1001]]) -> i64 {
            if a.len() < 3 { return 1 }
            let (l, r): (Vec<_>, Vec<_>) = a[1..].iter().copied().partition(|&x| x < a[0]);
            f(&l, c) * f(&r, c) % M * c[a.len() - 1][l.len()] % M
        }
        ((f(&n, &c) + M - 1) % M) as i32
    }
```

