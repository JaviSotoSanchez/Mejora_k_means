︠64696bf5-6b69-4718-a68a-26cf80c6be1cs︠
F.<x,y,z,a,b,c,d,a1,b1,c1,d1>= PolynomialRing(GF(11),'x,y,z,a,b,c,d,a1,b1,c1,d1')
︡2d1e663c-9555-4c4c-ad41-c989bae63a5e︡{"done":true}
︠0fa43169-37be-476c-b6b1-4bd8abbd112cs︠
L= a*x**3+b*x**2+c*x+d
M = a1*x**3+b1*x**2+c*x+d
︡0b969836-3aac-4ade-a876-2f0adf8f6253︡{"done":true}
︠4c5eb08f-bd24-4368-83a5-19fdfcf74cf8s︠
R = L.subs(x=y)*M-M.subs(x=y)*L
︡d6bd4027-802f-4e9e-ad1b-4e0f6aeff0de︡{"done":true}
︠92246f2a-d266-402f-bc44-1359d636e0bds︠
q,r= (R).quo_rem(x-y)
︡be058507-a974-49dc-a6dc-247a68e73536︡{"done":true}
︠d605347a-55a8-481c-a2af-e74b82481132s︠
q
︡722b2a62-50b2-4573-9a62-57dcd21f5514︡{"stdout":"x^2*y^2*b*a1 - x^2*y^2*a*b1 - x^2*y*a*c - x*y^2*a*c + x^2*y*c*a1 + x*y^2*c*a1 - x*y*b*c - x^2*a*d - x*y*a*d - y^2*a*d + x^2*d*a1 + x*y*d*a1 + y^2*d*a1 + x*y*c*b1 - x*b*d - y*b*d + x*d*b1 + y*d*b1\n"}︡{"done":true}
︠2d92a174-fed5-4725-96e6-ff6889976422s︠
r
︡fdc23219-e3c4-4369-b1d4-d7c659a85136︡{"stdout":"0\n"}︡{"done":true}
︠6f66210a-78a4-4e65-aea5-9f1ef2fda39es︠
h = q.homogenize(z)
︡69f9e026-4a4c-4f46-9e1f-f7bc88a1912c︡{"done":true}
︠c0266d59-cf60-4f35-96b9-80e9a605b3f4s︠
I = Ideal([F(h),F(h.derivative(z)),F(h.derivative(x)),F(h.derivative(y))])
︡b5944467-14ab-4fdb-b6ef-89aa2a09913c︡{"done":true}
︠4548248e-39a6-48cc-b3c9-5f0ece7b9e0bs︠
I.primary_decomposition()
︡12efd537-18d4-4582-8acd-bd17c862f1de︡{"stderr":"Error in lines 1-1\n"}︡{"stderr":"Traceback (most recent call last):\n  File \"/cocalc/lib/python2.7/site-packages/smc_sagews/sage_server.py\", line 1188, in execute\n    flags=compile_flags) in namespace, locals\n  File \"\", line 1, in <module>\n  File \"/ext/sage/sage-8.7_1804/local/lib/python2.7/site-packages/sage/rings/polynomial/multi_polynomial_ideal.py\", line 297, in __call__\n    return self.f(self._instance, *args, **kwds)\n  File \"/ext/sage/sage-8.7_1804/local/lib/python2.7/site-packages/sage/rings/polynomial/multi_polynomial_ideal.py\", line 853, in primary_decomposition\n    return [I for I, _ in self.complete_primary_decomposition(algorithm)]\n  File \"/ext/sage/sage-8.7_1804/local/lib/python2.7/site-packages/sage/rings/polynomial/multi_polynomial_ideal.py\", line 297, in __call__\n    return self.f(self._instance, *args, **kwds)\n  File \"/ext/sage/sage-8.7_1804/local/lib/python2.7/site-packages/sage/libs/singular/standard_options.py\", line 140, in wrapper\n    return func(*args, **kwds)\n  File \"/ext/sage/sage-8.7_1804/local/lib/python2.7/site-packages/sage/rings/polynomial/multi_polynomial_ideal.py\", line 774, in complete_primary_decomposition\n    P = primdecSY(self)\n  File \"sage/libs/singular/function.pyx\", line 1330, in sage.libs.singular.function.SingularFunction.__call__ (build/cythonized/sage/libs/singular/function.cpp:14947)\n    return call_function(self, args, ring, interruptible, attributes)\n  File \"sage/libs/singular/function.pyx\", line 1512, in sage.libs.singular.function.call_function (build/cythonized/sage/libs/singular/function.cpp:16771)\n    with opt_ctx: # we are preserving the global options state here\n  File \"sage/libs/singular/function.pyx\", line 1514, in sage.libs.singular.function.call_function (build/cythonized/sage/libs/singular/function.cpp:16683)\n    sig_on()\nKeyboardInterrupt\n"}︡{"done":true}
︠64a725eb-eac1-4ab5-ad09-d9327f52e495s︠
I.elimination_ideal([z,y])
︡142de940-1f4f-44e0-bf27-282967c53a42︡{"stdout":"Ideal (0) of Multivariate Polynomial Ring in x, y, z, a, b, c, d, a1, b1, c1, d1 over Finite Field of size 11"}︡{"stdout":"\n"}︡{"done":true}
︠f89003dd-8ad6-45f8-841e-b340a7bd2647s︠

h.factor()
︡78ff4760-f1b0-4ab2-8a4c-7d46c676712a︡{"stdout":"-x^2*y*z*a*c - x*y^2*z*a*c - x*y*z^2*b*c - x^2*z^2*a*d - x*y*z^2*a*d - y^2*z^2*a*d - x*z^3*b*d - y*z^3*b*d + x^2*y^2*b*a1 + x^2*y*z*c*a1 + x*y^2*z*c*a1 + x^2*z^2*d*a1 + x*y*z^2*d*a1 + y^2*z^2*d*a1 - x^2*y^2*a*b1 + x*y*z^2*c*b1 + x*z^3*d*b1 + y*z^3*d*b1\n"}︡{"done":true}
︠11d581aa-d2d9-4685-a8de-5836fcfcf169s︠
f
︡2ef82263-1e2c-476d-b421-1bc401ff781e︡{"stderr":"Error in lines 1-1\nTraceback (most recent call last):\n  File \"/cocalc/lib/python2.7/site-packages/smc_sagews/sage_server.py\", line 1188, in execute\n    flags=compile_flags) in namespace, locals\n  File \"\", line 1, in <module>\nNameError: name 'f' is not defined\n"}︡{"done":true}
︠69d5b5a6-9da9-4fb4-bf55-6656a31e4d36s︠
q.factor()
︡46674374-a697-4df6-8e22-9308b7366bf7︡{"stdout":"x^2*y^2*b*a1 - x^2*y^2*a*b1 - x^2*y*a*c - x*y^2*a*c + x^2*y*c*a1 + x*y^2*c*a1 - x*y*b*c - x^2*a*d - x*y*a*d - y^2*a*d + x^2*d*a1 + x*y*d*a1 + y^2*d*a1 + x*y*c*b1 - x*b*d - y*b*d + x*d*b1 + y*d*b1\n"}︡{"done":true}
︠7120d5ec-fdeb-45b5-844d-3efeeb47e4efs︠
q= q.subs(d1=0)
︡4dccf229-7de7-4288-ba60-c5779f3e89ef︡{"done":true}
︠98f0fd54-13e8-4008-9713-4faadbf5533bs︠
q = q.subs(d=0)
︡e37edd06-9775-443b-b0b9-97c3de5b53ba︡{"done":true}
︠51fd65ab-cf84-422d-8212-ab00608b6370s︠
q.factor()
︡2d4082f2-915a-44a6-8de5-b0922567a9f3︡{"stdout":"y * x * (x*y*b*a1 - x*y*a*b1 - x*a*c - y*a*c + x*c*a1 + y*c*a1 - b*c + c*b1)\n"}︡{"done":true}
︠3be86e3a-c913-4b34-bd83-4bfb53a07d1e︠









