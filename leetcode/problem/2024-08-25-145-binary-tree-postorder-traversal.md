---
layout: leetcode-entry
title: "145. Binary Tree Postorder Traversal"
permalink: "/leetcode/problem/2024-08-25-145-binary-tree-postorder-traversal/"
leetcode_ui: true
entry_slug: "2024-08-25-145-binary-tree-postorder-traversal"
---

[145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/description/) easy
[blog post](https://leetcode.com/problems/binary-tree-postorder-traversal/solutions/5687518/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25082024-145-binary-tree-postorder?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wpQtOupMIrU)
![1.webp](/assets/leetcode_daily_images/1c6339c0.webp)

#### Problem TLDR

Postorder tree traversal #easy #binary_tree

#### Intuition

Postorder is: left, right, current.

#### Approach

* let's reuse the method signature

#### Complexity

- Time complexity:
$$O(n^2)$$ for the list concatenation, $$O(n)$$ for Rust as it is optimizes recursion and concatenations

Kotlin runtime for a full binary tree with different depths:

```c
Depth	Nodes	Time (ms)
---------------------------
10	1023	1
11	2047	0
12	4095	2
13	8191	2
14	16383	3
15	32767	7
16	65535	16
17	131071	23
18	262143	44
19	524287	77
20	1048575	178
21	2097151	342
22	4194303	848
23	8388607	3917
```

For Rust:
```c
Depth   Nodes   Time (ms)
---------------------------
1       1       0.00
2       3       0.00
3       7       0.00
4       15      0.00
5       31      0.00
6       63      0.00
7       127     0.01
8       255     0.01
9       511     0.03
10      1023    0.04
11      2047    0.11
12      4095    0.19
13      8191    0.38
14      16383   0.76
15      32767   1.29
16      65535   2.68
17      131071  5.46
18      262143  14.94
19      524287  32.28
20      1048575 67.25
21      2097151 141.33
22      4194303 258.15
23      8388607 534.31
24      16777215        1057.31
25      33554431        2145.27
26      67108863        4266.18
27      134217727       8957.01
28      268435455       16987.34
```

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun postorderTraversal(root: TreeNode?): List<Int> = root?.run {
        postorderTraversal(left) +
        postorderTraversal(right) + listOf(`val`) } ?: listOf()

```
```rust

    pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        root.as_ref().map_or(vec![], |r| { let r = r.borrow();
            [&Self::postorder_traversal(r.left.clone())[..],
             &Self::postorder_traversal(r.right.clone())[..], &[r.val]].concat()
        })
    }

```

