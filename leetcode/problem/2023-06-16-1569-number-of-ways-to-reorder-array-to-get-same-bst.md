---
layout: leetcode-entry
title: "1569. Number of Ways to Reorder Array to Get Same BST"
permalink: "/leetcode/problem/2023-06-16-1569-number-of-ways-to-reorder-array-to-get-same-bst/"
leetcode_ui: true
entry_slug: "2023-06-16-1569-number-of-ways-to-reorder-array-to-get-same-bst"
---

[1569. Number of Ways to Reorder Array to Get Same BST](https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/description/) hard
[blog post](https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/solutions/3643907/kotlin-build-tree-count-permuts/)
[substack](https://dmitriisamoilenko.substack.com/p/16062023-1569-number-of-ways-to-reorder?sd=pf)
![image.png](/assets/leetcode_daily_images/6d3cd293.webp)

#### Join me on Telegram Leetcode_daily
https://t.me/leetcode_daily_unstoppable/247

#### Problem TLDR
Count permutations of an array with identical Binary Search Tree
#### Intuition
First step is to build a Binary Search Tree by adding the elements one by one.
Let's observe what enables the permutations in `[34512]`:
![image.png](/assets/leetcode_daily_images/f6d568a4.webp)
Left child `[12]` don't have permutations, as `1` must be followed by `2`. Same for the right `[45]`. However, when we're merging left and right, they can be merged in different positions.
Let's observe the pattern for merging `ab` x `cde`, `ab` x `cd`, `ab` x `c`, `a` x `b`:
![image.png](/assets/leetcode_daily_images/53428cc4.webp)
And another, `abc` x `def`:
![image.png](/assets/leetcode_daily_images/86f9a7ea.webp)
For each `length` of a left `len1` and right `len2` subtree, we can derive the equation for permutations `p`:
$$
p(len1, len2) = p(len1 - 1, len2) + p(len1, len2 - 1)
$$
Also, when left or right subtree have several permutations, like `abc`, `acb`, `cab`, and right `def`, `dfe`, the result will be multiplied `3 x 2`.

#### Approach
Build the tree, then compute the `p = left.p * right.p * p(left.len, right.len)` in a DFS.
#### Complexity
- Time complexity:
$$O(n^2)$$, n for tree walk, and n^2 for `f`
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

class Node(val v: Int, var left: Node? = null, var right: Node? = null)
data class R(val perms: Long, val len: Long)
fun numOfWays(nums: IntArray): Int {
    val mod = 1_000_000_007L
    var root: Node? = null
    fun insert(n: Node?, v: Int): Node {
        if (n == null) return Node(v)
        if (v > n.v) n.right = insert(n.right, v)
        else n.left = insert(n.left, v)
        return n
    }
    nums.forEach { root = insert(root, it) }
    val cache = mutableMapOf<Pair<Long, Long>, Long>()
    fun f(a: Long, b: Long): Long {
        return if (a < b) f(b, a) else if (a <= 0 || b <= 0) 1 else cache.getOrPut(a to b) {
            (f(a - 1, b) + f(a, b - 1)) % mod
        }
    }
    fun perms(a: R, b: R): Long {
        val perms = (a.perms * b.perms) % mod
        return (perms * f(a.len , b.len)) % mod
    }
    fun dfs(n: Node?): R {
        if (n == null) return R(1, 0)
        val left = dfs(n.left)
        val right = dfs(n.right)
        return R(perms(left, right), left.len + right.len + 1)
    }
    val res = dfs(root)?.perms?.dec() ?: 0
    return (if (res < 0) res + mod else res).toInt()
}

```

